// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/executable.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/metal/pipeline_layout.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/metal_executable_def_reader.h"
#include "iree/schemas/metal_executable_def_verifier.h"

typedef struct iree_hal_metal_executable_t {
  // Abstract resource used for injecting reference counting and vtable; must be at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // All loaded/compiled libraries.
  iree_host_size_t library_count;
  id<MTLLibrary>* libraries;

  // TODO(#18154): simplify struct per export.
  iree_host_size_t export_count;
  iree_hal_metal_kernel_params_t exports[];
} iree_hal_metal_executable_t;

static const iree_hal_executable_vtable_t iree_hal_metal_executable_vtable;

static iree_hal_metal_executable_t* iree_hal_metal_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_executable_vtable);
  return (iree_hal_metal_executable_t*)base_value;
}

static const iree_hal_metal_executable_t* iree_hal_metal_executable_const_cast(
    const iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_executable_vtable);
  return (const iree_hal_metal_executable_t*)base_value;
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during runtime.
//
// There are still some conditions we must be aware of (such as omitted names on functions with
// internal linkage), however we shouldn't need to bounds check anything within the flatbuffer
// after this succeeds.
static iree_status_t iree_hal_metal_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer data is not present or less than 16 bytes (%zu total)",
                            flatbuffer_data.data_length);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds and that we can
  // safely walk the file, but not that the actual contents of the flatbuffer meet our expectations.
  int verify_ret = iree_hal_metal_ExecutableDef_verify_as_root(flatbuffer_data.data,
                                                               flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_metal_ExecutableDef_table_t executable_def =
      iree_hal_metal_ExecutableDef_as_root(flatbuffer_data.data);

  // TODO(#18154): drop legacy flatbuffers compatibility.
  iree_hal_metal_ExportDef_vec_t exports_vec =
      iree_hal_metal_ExecutableDef_exports_get(executable_def);
  if (exports_vec != NULL) {
    iree_hal_metal_LibraryDef_vec_t libraries_vec =
        iree_hal_metal_ExecutableDef_libraries_get(executable_def);

    for (size_t i = 0; i < iree_hal_metal_ExportDef_vec_len(exports_vec); ++i) {
      iree_hal_metal_ExportDef_table_t export_def = iree_hal_metal_ExportDef_vec_at(exports_vec, i);
      uint32_t library_ordinal = iree_hal_metal_ExportDef_library_ordinal(export_def);
      if (library_ordinal >= iree_hal_metal_LibraryDef_vec_len(libraries_vec)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "export %" PRIhsz " references invalid library ordinal %u", i,
                                library_ordinal);
      }

      if (!flatbuffers_string_len(iree_hal_metal_ExportDef_entry_point_get(export_def))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "export %" PRIhsz " has no string identifier specified", i);
      }

      uint32_t constant_count = iree_hal_metal_ExportDef_constant_count_get(export_def);
      if (constant_count > IREE_HAL_METAL_MAX_PUSH_CONSTANT_COUNT) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "dispatch requiring %u constants exceeds limit of %d",
                                constant_count, IREE_HAL_METAL_MAX_PUSH_CONSTANT_COUNT);
      }

      iree_hal_metal_BindingBits_vec_t binding_flags_vec =
          iree_hal_metal_ExportDef_binding_flags_get(export_def);
      size_t binding_count = iree_hal_metal_BindingBits_vec_len(binding_flags_vec);
      if (binding_count > IREE_HAL_METAL_MAX_DESCRIPTOR_SET_BINDING_COUNT) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "dispatch requiring %" PRIhsz " bindings exceeds limit of %d",
                                binding_count, IREE_HAL_METAL_MAX_DESCRIPTOR_SET_BINDING_COUNT);
      }
    }
  } else {
    flatbuffers_string_vec_t entry_points_vec =
        iree_hal_metal_ExecutableDef_entry_points_get(executable_def);
    size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
    for (size_t i = 0; i < entry_point_count; ++i) {
      if (!flatbuffers_string_len(flatbuffers_string_vec_at(entry_points_vec, i))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "executable entry point %zu has no name", i);
      }
    }

    iree_hal_metal_ThreadgroupSize_vec_t threadgroup_sizes_vec =
        iree_hal_metal_ExecutableDef_threadgroup_sizes(executable_def);
    size_t threadgroup_size_count = iree_hal_metal_ThreadgroupSize_vec_len(threadgroup_sizes_vec);
    if (!threadgroup_size_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no threadgroup sizes present");
    }

    if (entry_point_count != threadgroup_size_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "entry points (%zu) and thread group sizes (%zu) count mismatch",
                              entry_point_count, threadgroup_size_count);
    }

    flatbuffers_string_vec_t shader_libraries_vec =
        iree_hal_metal_ExecutableDef_shader_libraries_get(executable_def);
    size_t shader_library_count = flatbuffers_string_vec_len(shader_libraries_vec);
    for (size_t i = 0; i < shader_library_count; ++i) {
      if (!flatbuffers_string_len(flatbuffers_string_vec_at(shader_libraries_vec, i))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "executable shader library %zu is empty", i);
      }
    }
    if (shader_library_count != 0 && entry_point_count != shader_library_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "entry points (%zu) and source libraries (%zu) count mismatch",
                              entry_point_count, shader_library_count);
    }

    flatbuffers_string_vec_t shader_sources_vec =
        iree_hal_metal_ExecutableDef_shader_sources_get(executable_def);
    size_t shader_source_count = flatbuffers_string_vec_len(shader_sources_vec);
    for (size_t i = 0; i < shader_source_count; ++i) {
      if (!flatbuffers_string_len(flatbuffers_string_vec_at(shader_sources_vec, i))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "executable shader source %zu is empty", i);
      }
    }

    if (shader_source_count != 0 && entry_point_count != shader_source_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "entry points (%zu) and source strings (%zu) count mismatch",
                              entry_point_count, shader_source_count);
    }

    if (!shader_library_count && !shader_source_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing shader library or source strings");
    }
  }

  return iree_ok_status();
}

// Returns an invalid argument status with proper Metal NSError annotations during compute pipeline
// creation.
static iree_status_t iree_hal_metal_get_invalid_kernel_status(const char* iree_error_template,
                                                              const char* metal_error_template,
                                                              NSError* ns_error,
                                                              iree_string_view_t entry_point,
                                                              const char* shader_source) {
  iree_status_t status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, iree_error_template);
  const char* ns_c_error = [ns_error.localizedDescription
      cStringUsingEncoding:[NSString defaultCStringEncoding]];  // autoreleased
  status = iree_status_annotate_f(status, metal_error_template, ns_c_error);
  if (shader_source) {
    return iree_status_annotate_f(status, "for entry point '%.*s' in MSL source:\n%s\n",
                                  (int)entry_point.size, entry_point.data, shader_source);
  }
  return iree_status_annotate_f(status, "for entry point '%.*s' in MTLLibrary\n",
                                (int)entry_point.size, entry_point.data);
}

// Compiles the given |entry_point| in the MSL |source_code| into MTLLibrary and writes to
// |out_library|. The caller should release |out_library| after done.
iree_status_t iree_hal_metal_compile_msl(iree_string_view_t source_code,
                                         iree_string_view_t entry_point, id<MTLDevice> device,
                                         MTLCompileOptions* compile_options,
                                         id<MTLLibrary>* out_library) {
  @autoreleasepool {
    NSError* error = nil;
    NSString* shader_source =
        [[[NSString alloc] initWithBytes:source_code.data
                                  length:source_code.size
                                encoding:[NSString defaultCStringEncoding]] autorelease];
    *out_library = [device newLibraryWithSource:shader_source
                                        options:compile_options
                                          error:&error];  // +1
    if (IREE_UNLIKELY(*out_library == nil)) {
      return iree_hal_metal_get_invalid_kernel_status(
          "failed to create MTLLibrary from shader source",
          "when creating MTLLibrary with NSError: %.*s", error, entry_point, source_code.data);
    }
  }

  return iree_ok_status();
}

// Compiles the given |entry_point| in the MSL library |source_data| into MTLLibrary and writes to
// |out_library|. The caller should release |out_library| after done.
static iree_status_t iree_hal_metal_load_mtllib(iree_const_byte_span_t source_data,
                                                iree_string_view_t entry_point,
                                                id<MTLDevice> device, id<MTLLibrary>* out_library) {
  @autoreleasepool {
    NSError* error = nil;
    dispatch_data_t data = dispatch_data_create(source_data.data, source_data.data_length,
                                                /*queue=*/NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    *out_library = [device newLibraryWithData:data error:&error];  // +1
    if (IREE_UNLIKELY(*out_library == nil)) {
      return iree_hal_metal_get_invalid_kernel_status(
          "failed to create MTLLibrary from shader source",
          "when creating MTLLibrary with NSError: %s", error, entry_point, NULL);
    }
  }

  return iree_ok_status();
}

// Creates MTL compute pipeline objects for the given |entry_point| in |library| and writes to
// |out_function| and |out_pso|. The caller should release |out_function| and |out_pso| after done.
static iree_status_t iree_hal_metal_create_pipeline_object(
    id<MTLLibrary> library, iree_string_view_t entry_point, const char* source_code,
    id<MTLDevice> device, id<MTLFunction>* out_function, id<MTLComputePipelineState>* out_pso) {
  @autoreleasepool {
    NSError* error = nil;
    NSString* function_name =
        [[[NSString alloc] initWithBytes:entry_point.data
                                  length:entry_point.size
                                encoding:[NSString defaultCStringEncoding]] autorelease];
    *out_function = [library newFunctionWithName:function_name];  // +1
    if (IREE_UNLIKELY(*out_function == nil)) {
      return iree_hal_metal_get_invalid_kernel_status("cannot find entry point in shader source",
                                                      "when creating MTLFunction with NSError: %s",
                                                      error, entry_point, source_code);
    }

    // TODO(#14047): Enable async pipeline creation at runtime.
    *out_pso = [device newComputePipelineStateWithFunction:*out_function error:&error];  // +1
    if (IREE_UNLIKELY(*out_pso == nil)) {
      [*out_function release];
      return iree_hal_metal_get_invalid_kernel_status(
          "invalid shader source", "when creating MTLComputePipelineState with NSError: %s", error,
          entry_point, source_code);
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_metal_compile_msl_and_create_pipeline_object(
    iree_string_view_t source_code, iree_string_view_t entry_point, id<MTLDevice> device,
    MTLCompileOptions* compile_options, id<MTLLibrary>* out_library, id<MTLFunction>* out_function,
    id<MTLComputePipelineState>* out_pso) {
  IREE_RETURN_IF_ERROR(
      iree_hal_metal_compile_msl(source_code, entry_point, device, compile_options, out_library));
  return iree_hal_metal_create_pipeline_object(*out_library, entry_point, source_code.data, device,
                                               out_function, out_pso);
}

iree_status_t iree_hal_metal_executable_create(
    id<MTLDevice> device, const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_executable_t* executable = NULL;

  IREE_RETURN_IF_ERROR(
      iree_hal_metal_executable_flatbuffer_verify(executable_params->executable_data));

  iree_hal_metal_ExecutableDef_table_t executable_def =
      iree_hal_metal_ExecutableDef_as_root(executable_params->executable_data.data);
  iree_hal_metal_LibraryDef_vec_t libraries_vec =
      iree_hal_metal_ExecutableDef_libraries_get(executable_def);
  iree_hal_metal_ExportDef_vec_t exports_vec =
      iree_hal_metal_ExecutableDef_exports_get(executable_def);

  // Calculate the total number of characters across all entry point names. This is only required
  // when tracing so that we can store copies of the names as the flatbuffer storing the strings
  // may be released while the executable is still live.
  iree_host_size_t export_count = iree_hal_metal_ExportDef_vec_len(exports_vec);
  iree_host_size_t total_entry_point_name_chars = 0;
  IREE_TRACE({
    if (exports_vec) {
      for (size_t i = 0; i < export_count; ++i) {
        iree_hal_metal_ExportDef_table_t export_def =
            iree_hal_metal_ExportDef_vec_at(exports_vec, i);
        flatbuffers_string_t entry_point = iree_hal_metal_ExportDef_entry_point_get(export_def);
        total_entry_point_name_chars += flatbuffers_string_len(entry_point);
      }
    } else {
      flatbuffers_string_vec_t entry_points_vec =
          iree_hal_metal_ExecutableDef_entry_points_get(executable_def);
      export_count = flatbuffers_string_vec_len(entry_points_vec);
      for (iree_host_size_t i = 0; i < export_count; ++i) {
        const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
        total_entry_point_name_chars += flatbuffers_string_len(entry_name);
      }
    }
  });

  // Create the HAL executable with storage for dynamic arrays.
  iree_host_size_t library_count = iree_hal_metal_LibraryDef_vec_len(libraries_vec);
  iree_host_size_t total_size =
      export_count * sizeof(executable->exports[0]) + sizeof(*executable) +
      library_count * sizeof(executable->libraries[0]) + total_entry_point_name_chars;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  iree_hal_resource_initialize(&iree_hal_metal_executable_vtable, &executable->resource);
  executable->host_allocator = host_allocator;
  executable->library_count = library_count;
  executable->libraries = (id<MTLLibrary>*)((uint8_t*)executable + sizeof(*executable) +
                                            export_count * sizeof(executable->exports[0]));
  memset(executable->libraries, 0, library_count * sizeof(executable->libraries[0]));
  executable->export_count = export_count;
  memset(executable->exports, 0, export_count * sizeof(executable->exports[0]));
  IREE_TRACE(char* string_table_buffer = (char*)((uint8_t*)executable->libraries +
                                                 library_count * sizeof(executable->libraries[0])));

  // Try to load as Metal library first. Otherwise, compile each MSL source string into a
  // MTLLibrary and get the MTLFunction for the entry point to build the pipeline state object.
  // TODO(#14047): Enable async MSL compilation at runtime.
  MTLCompileOptions* compile_options = [MTLCompileOptions new];  // +1
  compile_options.languageVersion = MTLLanguageVersion3_0;

  iree_status_t status = iree_ok_status();
  if (libraries_vec) {
    for (iree_host_size_t i = 0; i < library_count; ++i) {
      iree_hal_metal_LibraryDef_table_t library_def =
          iree_hal_metal_LibraryDef_vec_at(libraries_vec, i);
      flatbuffers_string_t library_str = iree_hal_metal_LibraryDef_library_get(library_def);
      if (flatbuffers_string_len(library_str) > 0) {
        status = iree_hal_metal_load_mtllib(
            iree_make_const_byte_span(library_str, flatbuffers_string_len(library_str)),
            iree_string_view_empty(), device, &executable->libraries[i]);
      } else {
        flatbuffers_string_t source_str = iree_hal_metal_LibraryDef_source_get(library_def);
        status = iree_hal_metal_compile_msl(
            iree_make_string_view(source_str, flatbuffers_string_len(source_str)),
            iree_string_view_empty(), device, compile_options, &executable->libraries[i]);
      }
      if (!iree_status_is_ok(status)) break;
    }
  }

  if (iree_status_is_ok(status)) {
    if (exports_vec) {
      for (iree_host_size_t i = 0; i < export_count; ++i) {
        iree_hal_metal_ExportDef_table_t export_def =
            iree_hal_metal_ExportDef_vec_at(exports_vec, i);
        uint32_t library_ordinal = iree_hal_metal_ExportDef_library_ordinal(export_def);
        flatbuffers_string_t entry_point = iree_hal_metal_ExportDef_entry_point_get(export_def);

        iree_hal_metal_kernel_params_t* params = &executable->exports[i];
        params->library = executable->libraries[library_ordinal];
        [params->library retain];  // +1

        status = iree_hal_metal_create_pipeline_object(
            params->library,
            iree_make_string_view(entry_point, flatbuffers_string_len(entry_point)), NULL, device,
            &params->function, &params->pso);
        if (!iree_status_is_ok(status)) break;

        iree_hal_metal_ThreadgroupSize_struct_t threadgroup_size =
            iree_hal_metal_ExportDef_threadgroup_size_get(export_def);
        params->threadgroup_size[0] = threadgroup_size->x;
        params->threadgroup_size[1] = threadgroup_size->y;
        params->threadgroup_size[2] = threadgroup_size->z;

        params->constant_count = iree_hal_metal_ExportDef_constant_count_get(export_def);

        iree_hal_metal_BindingBits_vec_t binding_flags_vec =
            iree_hal_metal_ExportDef_binding_flags_get(export_def);
        size_t binding_count = iree_hal_metal_BindingBits_vec_len(binding_flags_vec);
        params->binding_count = binding_count;
        for (size_t j = 0; j < binding_count; ++j) {
          iree_hal_metal_BindingBits_enum_t flags =
              iree_hal_metal_BindingBits_vec_at(binding_flags_vec, j);
          uint64_t binding_bit = 1ull << j;
          if (iree_all_bits_set(flags, iree_hal_metal_BindingBits_READ_ONLY)) {
            params->binding_flags.read_only |= binding_bit;
          }
          if (iree_all_bits_set(flags, iree_hal_metal_BindingBits_INDIRECT)) {
            params->binding_flags.indirect |= binding_bit;
          }
        }

        // Stash the entry point name in the string table for use when tracing.
        IREE_TRACE({
          iree_host_size_t entry_name_length = flatbuffers_string_len(entry_point);
          memcpy(string_table_buffer, entry_point, entry_name_length);
          params->function_name = iree_make_string_view(string_table_buffer, entry_name_length);
          string_table_buffer += entry_name_length;
        });
      }
    } else {
      flatbuffers_string_vec_t entry_points_vec =
          iree_hal_metal_ExecutableDef_entry_points_get(executable_def);
      iree_hal_metal_ThreadgroupSize_vec_t threadgroup_sizes_vec =
          iree_hal_metal_ExecutableDef_threadgroup_sizes(executable_def);
      flatbuffers_string_vec_t shader_libraries_vec =
          iree_hal_metal_ExecutableDef_shader_libraries_get(executable_def);
      flatbuffers_string_vec_t shader_sources_vec =
          iree_hal_metal_ExecutableDef_shader_sources_get(executable_def);
      size_t shader_library_count = flatbuffers_string_vec_len(shader_libraries_vec);
      size_t shader_source_count = flatbuffers_string_vec_len(shader_sources_vec);
      for (size_t i = 0, e = iree_max(shader_library_count, shader_source_count); i < e; ++i) {
        id<MTLLibrary> library = nil;
        id<MTLFunction> function = nil;
        id<MTLComputePipelineState> pso = nil;

        flatbuffers_string_t source_code = NULL;
        flatbuffers_string_t entry_point = flatbuffers_string_vec_at(entry_points_vec, i);
        iree_string_view_t entry_point_view =
            iree_make_string_view(entry_point, flatbuffers_string_len(entry_point));

        if (shader_library_count != 0) {
          flatbuffers_string_t source_library = flatbuffers_string_vec_at(shader_libraries_vec, i);
          status = iree_hal_metal_load_mtllib(
              iree_make_const_byte_span(source_library, flatbuffers_string_len(source_library)),
              entry_point_view, device, &library);
        } else {
          source_code = flatbuffers_string_vec_at(shader_sources_vec, i);
          status = iree_hal_metal_compile_msl(
              iree_make_string_view(source_code, flatbuffers_string_len(source_code)),
              entry_point_view, device, compile_options, &library);
        }
        if (!iree_status_is_ok(status)) break;

        status = iree_hal_metal_create_pipeline_object(library, entry_point_view, source_code,
                                                       device, &function, &pso);
        if (!iree_status_is_ok(status)) break;

        // Package required parameters for kernel launches for each entry point.
        iree_hal_metal_kernel_params_t* params = &executable->exports[i];
        params->library = library;
        params->function = function;
        params->pso = pso;
        params->threadgroup_size[0] = threadgroup_sizes_vec[i].x;
        params->threadgroup_size[1] = threadgroup_sizes_vec[i].y;
        params->threadgroup_size[2] = threadgroup_sizes_vec[i].z;
        params->layout = executable_params->pipeline_layouts[i];
        iree_hal_pipeline_layout_retain(params->layout);

        params->constant_count = iree_hal_metal_pipeline_layout_push_constant_count(params->layout);

        const iree_hal_descriptor_set_layout_t* set0_layout =
            iree_hal_metal_pipeline_layout_descriptor_set_layout(params->layout, 0);
        params->binding_count = iree_hal_metal_descriptor_set_layout_binding_count(set0_layout);
        for (uint32_t j = 0; j < params->binding_count; ++j) {
          const iree_hal_descriptor_set_layout_binding_t* binding =
              iree_hal_metal_descriptor_set_layout_binding(set0_layout, j);
          if (iree_all_bits_set(binding->flags, IREE_HAL_DESCRIPTOR_FLAG_READ_ONLY)) {
            params->binding_flags.read_only = 1ull << j;
          }
        }

        // Stash the entry point name in the string table for use when tracing.
        IREE_TRACE({
          iree_host_size_t entry_name_length = flatbuffers_string_len(entry_point);
          memcpy(string_table_buffer, entry_point, entry_name_length);
          params->function_name = iree_make_string_view(string_table_buffer, entry_name_length);
          string_table_buffer += entry_name_length;
        });
      }
    }
  }

  [compile_options release];  // -1

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_executable_destroy(iree_hal_executable_t* base_executable) {
  iree_hal_metal_executable_t* executable = iree_hal_metal_executable_cast(base_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->export_count; ++i) {
    iree_hal_metal_kernel_params_t* entry_point = &executable->exports[i];
    [entry_point->pso release];       // -1
    [entry_point->function release];  // -1
    [entry_point->library release];   // -1
    iree_hal_pipeline_layout_release(entry_point->layout);
  }

  for (iree_host_size_t i = 0; i < executable->library_count; ++i) {
    id<MTLLibrary> library = executable->libraries[i];
    [library release];  // -1
  }

  iree_allocator_free(executable->host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_metal_executable_entry_point_kernel_params(
    const iree_hal_executable_t* base_executable, int32_t entry_point,
    iree_hal_metal_kernel_params_t* out_params) {
  const iree_hal_metal_executable_t* executable =
      iree_hal_metal_executable_const_cast(base_executable);
  if (entry_point >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "invalid entry point ordinal %d",
                            entry_point);
  }
  memcpy(out_params, &executable->exports[entry_point], sizeof(*out_params));
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t iree_hal_metal_executable_vtable = {
    .destroy = iree_hal_metal_executable_destroy,
};
