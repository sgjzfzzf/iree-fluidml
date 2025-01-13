# AMD GPU HAL Driver (`amdgpu`)

TODO(benvanik): document design.

-DIREE_BUILD_COMPILER=ON
-DIREE_TARGET_BACKEND_ROCM=ON

-DIREE_HAL_DRIVER_AMDGPU=ON

-DIREE_HIP_TEST_TARGET_CHIP=gfx1100


## Engineering Notes

### HSA/ROCR Dependencies

TODO

### Physical Device Grouping

Much of the HAL implementation requires that all physical devices participating as part of a single logical device are "compatible." This allows attributes like queue size limits, allocation granularities, and supported ISAs to be assumed uniform. If at some point we want to allow multiple devices with differing attributes we could do so as it makes sense. As an example supporting devices with different ISAs is a small extension to select and load the appropriate per-device device library binary instead of loading it once and reusing it. With queue affinity passed on HAL calls we can also support things like unified memory import when the devices the imported buffer is requested to be used on all support unified memory even if other physical devices in the group do not.

### Host and Device Memory

Though HSA allows pointers to be accessed from the host and any device there is a non-trivial performance impact of actually accessing across devices. Tightly tuned access code that is able to be pipelined by the host or device is sometimes OK but any data dependencies across the bus can total microseconds of delay. For example if the device needs to access a data structure and must first query the pointer of the data structure from another data structure that is stored on the host the single access can take 1-10 microseconds if uncached. Our device library code runs infrequently (between tens to thousands of dispatches) and is almost guaranteed to have a cold cache which leads to expensive stalls on cache fills for the dependent data. The reverse applies to the host: device library maintained data structures such as queue entries are expensive to access from the host and always result in cache misses.

Host HAL implementation code that retains pointers to device memory should annotate any pointer that _may_ be in device memory with `IREE_AMDGPU_DEVICE_PTR`. This is a hint to the reader that indirecting through the pointer or accessing the data referenced by it may incur significant overhead. The primary pattern that requires careful attention is indirection through remote memory (host doing `some_device_ptr->other_device_ptr->value`) and in such cases we try to shadow indirected values (host doing `host_shadow_ptr->other_device_ptr->value`) or avoid dependent calculations (embedding data in structs such that the remote side need only do pointer math). A rarer pattern but one that can have even higher performance impact is RPCs where for example the device side originates a request the host side must respond to: in these cases the information that was already in device registers in order to produce the request must make it all the way to the host - if that requires ~10 cache misses walking data structures it can easily add 10's of microseconds of latency. In such cases we try to encode all of the required information in AQL packets such that single cache line fills are sufficient to avoid any dependent indirection in hot paths.

#### Memory Visibility

Allowing access to memory located on one device to another device is expensive enough to want to avoid however once access is allowed there's no measurable performance penalty for having the access shared. Performance implications arise when the memory is accessed from remote devices as there's additional bus traffic involved in each cache line fill. Our usage of cross-device memory in the HAL implementation is limited mostly to AQL queues and our custom queue data structures as those are used for RPC. Particular cases where we replicate data to ensure local access even though the memory may be easy to make accessible are hot data structures like command buffers and kernel descriptors: in such cases we want to ensure that issuing commands and dispatches are guaranteed to be local (if not cached) to the device performing the work.

#### Queue Placement

**WARNING: Experimental**

ROCR recently grew the ability to allocate queue ringbuffers in device memory instead of the default host memory via the `HSA_ALLOCATE_QUEUE_DEV_MEM` environment variable. Work is ongoing to make it part of the API so we can adjust it per-queue and store the queue structure (read/write packet IDs, etc) in device memory as well: https://github.com/ROCm/ROCR-Runtime/issues/269

Given that 99% of all of our AQL queue submissions are from the device command buffer execution and they are going to the same device we need the execution queues to be in device memory. Today we go across the system bus for every single packet insertion and that adds significant latency.

For queues submissions that happen from the host there may still be value in keeping them in host memory assuming that the firmware and hardware are designed to properly prefetch. Early testing shows that it's not particularly well optimized and that any microseconds we save on the host end up getting directly transferred to the device.

### Host HAL

#### Blocked Allocation

* See: [iree/hal/drivers/amdgpu/util/block_pool.h](/runtime/src/iree/hal/amdgpu/util/block_pool.h)

Allocations serviced by HSA (or the kernel) can be extremely expensive by involving syscalls, page table updates, full cache flushes, and sometimes full device synchronization. The total number of allocations also has scaling limits due to code on the allocation path performing linear scans or queries across existing allocations. To control for this we have our own suballocator that is backed by fixed-size device memory blocks. Each block is sized at the recommended allocation granularity of the device and accessible to all devices that may need to access it. By using fixed-size blocks that we infrequently allocate or free (usually only on trims) we avoid introducing a significant amount of fragmentation beyond what the application running on top of the HAL introduces by its own usage. Many subsystems within the HAL share the block pools and we bucket between small and large allocations to avoid internal fragmentation within the blocks.

#### Buffer and Semaphore Pooling

* See: [iree/hal/drivers/amdgpu/buffer_pool.h](/runtime/src/iree/hal/amdgpu/buffer_pool.h)
* See: [iree/hal/drivers/amdgpu/semaphore_pool.h](/runtime/src/iree/hal/amdgpu/semaphore_pool.h)

Internal buffers and semaphores (those created and used exclusively within the AMDGPU HAL) have device-side metadata structures backing the host-side HAL `iree_hal_buffer_t` and `iree_hal_semaphore_t`. Since buffers and semaphores may be used across any physical device within a logical device group both pool types are shared across all physical devices. External HAL buffers and semaphores (those created by another HAL implementation or wrapping host resources) are not able to be accessed directly by the device and are not pooled.

#### Command Buffer Recording

* See: [iree/hal/drivers/amdgpu/command_buffer.h](/runtime/src/iree/hal/amdgpu/command_buffer.h)
* See: [iree/hal/drivers/amdgpu/device/command_buffer.h](/runtime/src/iree/hal/amdgpu/device/command_buffer.h)

TODO

#### Worker Service

* See: [iree/hal/drivers/amdgpu/host_worker.h](/runtime/src/iree/hal/amdgpu/host_worker.h)
* See: [iree/hal/drivers/amdgpu/device/host.h](/runtime/src/iree/hal/amdgpu/device/host.h)

TODO

#### Tracing

* See: [iree/hal/drivers/amdgpu/trace_buffer.h](/runtime/src/iree/hal/amdgpu/trace_buffer.h)
* See: [iree/hal/drivers/amdgpu/device/tracing.h](/runtime/src/iree/hal/amdgpu/device/tracing.h)

TODO: ringbuffer
TODO: macros
TODO: flushing and timing correlation

### Device Library

* See: [build_tools/bazel/iree_amdgpu_binary.bzl](/build_tools/bazel/iree_amdgpu_binary.bzl)
* See: [build_tools/cmake/iree_amdgpu_binary.cmake](/build_tools/cmake/iree_amdgpu_binary.cmake)

Our device library containing the scheduler and all builtin kernels used by it is written in bare-metal C23. We do not use a libc-alike library or the AMD device libraries. Building device library binaries requires only a clang/llvm-link/lld with the AMDGPU targets enabled.

Because we are developing bare-metal style for a GPU nearly everything above the base language level is off the table - no TLS (what would that even mean?), no globals, no library or system calls, and no C++ things (e.g. global initializers). Since even things like atomics differ external header-only libraries are unlikely to be usable unless very tightly scoped and we'd fork them for internal use if needed.

The device library code living under [iree/hal/drivers/amdgpu/device/](/runtime/src/iree/hal/amdgpu/device/) depends wholely on itself. The host HAL implementation in the parent [iree/hal/drivers/amdgpu/](/runtime/src/iree/hal/amdgpu/) pulls in headers from the device library in order to share data structures and enums but does not try to compile the code. Code specific to the device library is guarded by `IREE_AMDGPU_TARGET_DEVICE` and code specific to the host compilation environment is guarded by `IREE_AMDGPU_TARGET_HOST`. Since crossing the streams can quickly spiral into madness effort is spent to avoid intermixing as much as practical while trying to minimize duplication of things that may potentially get out of sync.

#### Versioning

The device library is only ever shipped with the runtime it is built for. The API between host and device is not stable and considered an implementation detail of the HAL. If any of the API does leak out into deployments that may have different versions - such as compiled kernels that may rely on custom ABI or details of our execution environment - they will need to be versioned appropriately.

#### Architecture-specific Binaries

* See: [iree/hal/drivers/amdgpu/device/BUILD.bazel](/runtime/src/iree/hal/drivers/amdgpu/device/BUILD.bazel)

Unfortunately all AMD device binaries are architecture dependent and not backward or forward compatible. This results in us needing to include precompiled binaries for every architecture that the runtime may be used with and there are quite a few. In development builds it's best to set the single architecture or two a developer is using to keep compile times down but release builds need to produce binaries for all officially supported architectures. In the future if AMD gains a forward-compatible representation we'll jump on that: nothing about our device library relies on architecture-specific features as it's mostly straight-line scalar C code running in a single work item shuffling bits around. Given growing SPIR-V support in LLVM that may be one route assuming that SPIR-V binaries can be loaded (and JITed) by the HSA code object mechanism.

**TODO(benvanik)**: document cmake flags for controling which binaries are built.

#### Scoped Atomics

* See: [iree/hal/drivers/amdgpu/device/support/common.h](/runtime/src/iree/hal/amdgpu/device/support/common.h)

Atomics performed on the device need to indicate the scope at which they are synchronized within the system. Normal C11 atomics do not include a scope and are assumed to operate at system level meaning that an atomic update of a value that is only ever produced and consumed on a single device must be made visible to the entire system (host and all other devices). To avoid this potential performance issue atomic operations that are used on devices generally include a scope that indicates the visibility and that scope must be as wide as required and should be as narrow as possible. The `iree_amdgpu_scoped_atomic_*` functions mirror the C11 atomics but also take a scope, e.g. `iree_amdgpu_scoped_atomic_fetch_add(..., iree_amdgpu_memory_scope_system);` Where we know that a particular atomic operation _may_ cross device boundaries we err on the side of `iree_amdgpu_memory_scope_system` and when we are positive it _will not_ we use `iree_amdgpu_memory_scope_device`. The finer scopes such as `work_item` and `sub_group` are rarely used in our device library as we rarely use per-work-item atomics as part of a single dispatch.

#### HSA Queues and Signals

* See: [iree/hal/drivers/amdgpu/device/support/queue.h](/runtime/src/iree/hal/amdgpu/device/support/queue.h)
* See: [iree/hal/drivers/amdgpu/device/support/signal.h](/runtime/src/iree/hal/amdgpu/device/support/signal.h)

The HSA specification defines the `hsa_queue_t` structure and the interface for `hsa_signal_t`. Usage of these structures via `hsa_*` functions normally requires the AMD device library to be linked into the device library binary and is a subset of the operations we perform as (effectively) an HSA driver implementation. Thankfully the AMDGPU-specific definitions of the queue and signal (`amd_queue_t` and `amd_signal_t`) are specified and something we can (for our purposes) directly poke. We rely on ROCR to allocate queues and configure the driver and device but manipulate the queues directly (read/write packet IDs, the queue ringbuffer, etc). Signals are simpler and for the most part on architectures we support (everything in the last ~5+ years) they are simply atomic operations that are easy to implement directly. For device-only signals not backed by platform events we don't need to use ROCR and can instead treat any memory as a signal, enabling us to slab allocate thousands of signals cheaply and pool signals entirely on device (critical when timing dispatches that each require a unique signal).

Since the AMD device support library is not shipped as part of LLVM and the implementations of the functions are not required we redefine the `amd_queue_t` and `amd_signal_t` structures in our own headers. This avoids the need to fetch/link the library (which is geared for OpenCL and HIP and includes a significant amount of baggage) and avoid the requirement for implicit kernel arguments.

#### Buffers

* See: [iree/hal/drivers/amdgpu/device/buffer.h](/runtime/src/iree/hal/amdgpu/device/buffer.h)

Supporting replayable command buffers requires that buffers are referenced indirectly as the buffers used may change from submission to submission. The HAL is structured to allow this by supporting binding tables passed alongside command buffers when submitting work for execution but also allows for any buffer referenced statically to be indirected. The primary source of indirect buffers is queue-ordered allocation (`iree_hal_device_queue_alloca`) where the address of an allocation is not guaranteed to be available (let alone committed) until the asynchronous operation completes. To enable recording command buffers that reference such buffers or submitting command buffers for execution with binding tables containing the results of asynchronous allocations we use fat pointers throughout and try to perform the indirection just-in-time.

`iree_hal_amdgpu_device_buffer_ref_t` is used to reference buffers of all types with an enum indicating whether the pointer is available immediately (a statically allocated buffer), available with delay (a queue-ordered allocation), or sourced from a binding table slot at submission-time. Each buffer allocated in queue-order is assigned a `iree_hal_amdgpu_device_allocation_handle_t` in device memory that is returned immediately when requesting the allocation and that has a pointer which is updated when the allocation is committed.

Currently the implementation is performing the indirection on allocation handles to fetch the final address when either queue submissions (copy, fill, etc) or dispatches within a command buffer are issued. This avoids any additional overhead within the kernels as they just see a device pointer. It does mean that each binding is resolved per dispatch issued instead of per binding in the binding table as is the minimum required and that may be improved in the future at the cost of slightly higher latency when scheduling a command buffer.

#### Semaphores

* See: [iree/hal/drivers/amdgpu/semaphore.h](/runtime/src/iree/hal/amdgpu/semaphore.h)
* See: [iree/hal/drivers/amdgpu/device/semaphore.h](/runtime/src/iree/hal/amdgpu/device/semaphore.h)

The device library supports two types of semaphores: internal and external. Internal semaphores are those created by the AMDGPU HAL and used exclusively with it (on any device). Internal semaphores are strongly preferred and should be used in nearly all cases except when interacting with external APIs (interop with Vulkan, host code, etc).

Internal semaphores are backed by `iree_hal_amdgpu_device_semaphore_t` in device memory which maintains a list of waiters to be woken as certain semaphore values are reached. This allows the device to sequence work both on itself and across other devices or the host by waking the target without needing to involve the host. No platform events are involved and in the steady state of a device->device pipeline no host interrupts are required. To support the host waiting on internal semaphores signaled by a device we also include an HSA signal per semaphore that mirrors the payload value of the semaphore thereby allowing HSA multi-wait operations to efficiently block (or spin).

External semaphores are treated as opaque from the perspective of the device library and always route through the host when signaled. Waiting on external semaphores must happen on the host as the device cannot interact with the platform or vtabled HAL object.

#### Kernels

* See: [iree/hal/drivers/amdgpu/executable.h](/runtime/src/iree/hal/amdgpu/device/executable.h)
* See: [iree/hal/drivers/amdgpu/device/kernels.h](/runtime/src/iree/hal/amdgpu/device/kernels.h)
* See: [iree/hal/drivers/amdgpu/device/kernel_tables.h](/runtime/src/iree/hal/amdgpu/device/kernel_tables.h)

AQL dispatch packets require metadata (segment sizes, workgroup sizes, etc) and we retain these per-kernel in a device-local `iree_hal_amdgpu_device_kernel_args_t` structure. In addition to the attributes required by AQL we include ones used by the device library to issue the dispatches such as constant and binding count in order to decode command buffers and tracing identifiers used to tag trace events with a host-visible source location. HAL executables allocate kernel arguments on each device and populate them as defined in the metadata embedded in the executable binary. Built-in device library entry points are specified in a table baked into the binary.

Because every dispatch packet enqueued requires the metadata we need to store the information in device memory. In addition to the per-device copies the host also retains a copy in order to record and validate command buffers without hitting device memory. Note that all copies of the metadata between the host and devices a kernel is loaded on must be identical.

##### Implicit Arguments

OpenCL and HIP both tag on a non-trivial amount of implicit arguments to each kernel launch. These support device library features such as host calls (printf, etc), device-side enqueuing (in OpenCL), and programming model support (OpenCL grid offsets). We need none of those and as such as omit them from both our device library functions and compiler-produced kernel functions. This can save considerable kernarg ringbuffer space - over 1000 dispatches in a command buffer at 256 bytes per dispatch can easily cause stalls without overallocating the kernarg ring. This is one cause of host stalls observed when dispatching even moderate amount of work via HIP and OpenCL: the host spins waiting for more ringbuffer space. If we ever find ourselves needing these implicit arguments (for dispatching HIP or OpenCL kernels unmodified) we can enable conditional support for those dispatches that need them, however some features such as host support will not be practical to support in most cases.

#### Command Buffer Execution

* See: [iree/hal/drivers/amdgpu/command_buffer.h](/runtime/src/iree/hal/amdgpu/command_buffer.h)
* See: [iree/hal/drivers/amdgpu/device/command_buffer.h](/runtime/src/iree/hal/amdgpu/device/command_buffer.h)

Command buffers are recorded into an in-memory format optimized for issuing repeatedly by the device scheduler. Information that may change across submissions such as buffers referenced in the per-submission binding table or offsets in HSA queues or kernarg ringbuffers are left as symbolic during recording and populated by the device scheduler on-demand. The in-memory representation is a program with one or more basic blocks (`iree_hal_amdgpu_device_command_block_t`) containing one or more commands (`iree_hal_amdgpu_device_cmd_t`) terminating in a control command (e.g. `iree_hal_amdgpu_device_cmd_branch_t`). When a queue submission is ready to issue (all waits satisfied) the device-side scheduler uses the static command buffer information to populate an `iree_hal_amdgpu_device_execution_state_t` based on the submission pointing at the entry block and referencing reserved ranges of various ringbuffers and pools. A device-side block issue kernel is launched to then convert each command in the block from the in-memory representation to one or more AQL packets in the target execution queue. The commands then execute as the device processes the packets and when the terminating control command is reached (branch, return, etc) the next block or originating scheduler is enqueued to continue program execution or complete the submission.

Each submission maintains its own execution state allowing for the same command buffer recording to be issued on the same device simultaneously. The immutable in-memory representation of the command buffer encodes offsets/deltas for kernargs, completion signals, and tracing events that are overlayed on to the scheduled resources when the command buffer is issued. Since some of those offsets/deltas are submission-dependent or queue-dependent sometimes alternates are included in the metadata to allow the scheduler to pick the appropriate set (e.g. including per-command profiling query IDs for when operating at different tracing levels).

##### Indirect Dispatches

An indirect dispatch is one that sources its workgroup count from a buffer immediately prior to executing. This allows for prior dispatches within the same command buffer to produce the `uint32_t workgroup_count[3]` based on data-dependent information. In user programs with dynamically sized buffers (originating from dynamically shaped tensors in ML programs) this can often allow for the same command buffer to be replayed even if shapes vary. AQL does not currently have a way to perform this as part of dispatch packets, unfortunately, and we have to emulate it.

Emulation requires an ancillary dispatch that reads the `workgroup_count[3]` from the source buffer to patches the subsquent actual dispatch packet. This adds additional latency to the dispatch that would be avoided if the AQL packet natively supported a way of specifying a buffer; note the packet processor may incur memory fetch costs but those are orders of magnitude less costly than the emulation.

In the common case today indirect dispatches are frequently fixed at the time a command buffer is issued. To avoid the emulation overhead such dispatches are recorded with the `IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_STATIC` flag and the workgroup count is fetched as part of producing the AQL dispatch packet. Any indirect dispatches not known to be static during a submission instead use `IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_DYNAMIC` and get the additional dispatch to patch the packet.

It's a strong request for this to be improved in future packet processor versions - programs will only get more dynamic.

##### Large Command Buffer Optimization

Outside of specific usage patterns (e.g. layer-level tensor parallelism) most command buffers are on the order of 500 to 5000 commands. Our queue scheduling behavior allows that to be a single host->device or device->device operation regardless of count and reduce cross-device traffic considerably. Once issuing command buffer execution on the target device we are scoped to that local device and ordered in the target execution queue such that no synchronization (internal or otherwise) is required. As part of recording the command buffer all per-command information is encoded such that each command can be issued independently. This includes dynamic inputs like buffer bindings and kernarg ringbuffer offsets as well as things only known at issue time like the AQL packet range in the target execution queue.

The `iree_hal_amdgpu_device_cmd_block_issue` kernel is responsible for taking the immutable in-memory command buffer recording and translating each command into zero or more AQL packets. The scheduler reserves ranges of packets, signals, and kernargs before dispatching the kernel with one work item per command. Because the entire range of AQL packets is reserved the device is able to translate commands in any order as work items progress and immediately start processing. This relies on the AQL packet processing rules that say that a hardware doorbell only need indicate packets will have their type updated from `INVALID` to something valid _at some point_ and not that it needs to be prior to notifying the device.

An additional optimization not performed today would be sorting the commands within a block such that commands with the same type are contiguous. Alternatively an indirection table of a flattened contiguous space could be stored as part of the recorded command buffer. This would allow us to use a larger workgroup size knowing that each work item would be producing the same type of packet and therefore running (roughly) the same instructions.

##### Small Command Buffer Optimization

If performance is a conern as much work as possible must be recorded into the same command buffer as possible. Though the scheduling model allows us to avoid a tremendous amount of host->device traffic by requiring only a single host->device enqueue of the command buffer regardless of command count that only has impact at a scale above ~5 commands. Under that the additional dispatch latency of the scheduler and command buffer issue and retire logic outweighs the benefits of batching.

For cases where command buffers are tiny (1 or 2 commands) it's preferred that applications instead issue queue operations if possible (such as `iree_hal_device_queue_fill` and `iree_hal_device_queue_copy`) as this avoids the extra command buffer logic. The host-side HAL queue _could_ identify such command buffers and process them directly on the host but today we do not (it would require significant logic duplication).

In cases where queue operations are not available (such as dispatches) and the device needs to schedule a small command buffer we switch to issuing commands serially on the execution queue instead of dispatching the `iree_hal_amdgpu_device_cmd_block_issue` kernel to do so in parallel on the control queue. This trades off slower serial packet enqueuing against the extra dispatch latency of the issue.

This particular optimization can be useful even outside of issuing small command buffers as when a command buffer contains control flow commands (branch, etc) we may end up with some blocks that contain very few commands such as a loop body that contains only two dispatches. For conditional execution ("dispatch kernel A or kernel B based on this buffer value") it's preferred that indirect dispatch is used as a form of predication: not enqueuing a packet at issue time is significantly faster than jumping around between blocks within a command buffer.

#### Scheduler

* See: [iree/hal/drivers/amdgpu/queue.h](/runtime/src/iree/hal/amdgpu/queue.h)
* See: [iree/hal/drivers/amdgpu/device/scheduler.h](/runtime/src/iree/hal/amdgpu/device/scheduler.h)

```
                   +--------------+     +-----------+
+-------------+    |+--------------+    |+-----------+    +---------------+
| logical dev |--->+| physical dev |--->+| scheduler |-+->| control queue |
+-------------+     +--------------+     +-----------+ |  +---------------+
                                                       |  +-----------------+
                                                       |  |+-----------------+
                                                       +->+| execution queue |
                                                           +-----------------+
```

Each physical device may have one or more schedulers with an associated control queue and one or more execution queues. Each scheduler maintains its own execution resources such as kernarg storage and pending queue entry lists allowing each to run independently. The exact topology is configurable and may vary from one extreme (1:1:1:1 on small devices) to the other (2:8:8:16 on larger devices).

The control queue is exclusively for scheduler operations and runs with elevated priority. Scheduler operations are generally very short (microseconds) and what either issue new work for the device to process or retire completed work in order to unblock dependent work. Since dependent work may be on the host or another device it's important that the latency is as low in order to reduce bubbles. By using a single queue for scheduler-related work (such as command buffer execution management) we ensure only a single scheduler operation is running at a time and we need no internal synchronization. Note that operations on the control queue may still need to synchronize with the host or other devices via structured mechanisms (mailboxes/HSA queues/etc).

Schedulers obey the dependency requirements of submitted work as represented by semaphore wait and signal lists attached to each queue operation. The execution queues may be shared across multiple schedulers or exclusive to a particular scheduler: each command buffer execution can be thought of as a fiber with the hardware cooperatively scheduling the commands in each. A single scheduler may decide to run independent operations on separate execution queues to gain concurrency or dependent operations on the same queue to reduce overheads (as no synchronization is required within a single queue).

Today the scheduler is relatively basic: each time a scheduling operation is to be performed the `iree_hal_amdgpu_device_queue_scheduler_tick` kernel is dispatched on the control queue and it handles any scheduler operations that can be performed. By having the tick poll and process any work available at the time it is actually executed (vs enqueued) we avoid additional latency from dispatching per queue operation and waiting for each to clear. A single tick may accept incoming queue entries from the host or other another scheduler, retire completed entries, check whether they or any existing entry is ready to execute (all waiters satisfied), and issue all ready entries (if resources allow). In common cases of N chained queue entries - even if across multiple devices - this results in 2 + N scheduler ticks (initial issue, each retire->issue, and final retire).

##### Tracing

* See: [iree/hal/drivers/amdgpu/trace_buffer.h](/runtime/src/iree/hal/amdgpu/trace_buffer.h)
* See: [iree/hal/drivers/amdgpu/device/tracing.h](/runtime/src/iree/hal/amdgpu/device/tracing.h)

An AMDGPU-specific tracing mechanism very similar (but not identical to) the main IREE tracing mechanism is provided to instrument device library code and record performance information that can be relayed to host tracing tools.

Each device library kernel has through some means access to an `iree_hal_amdgpu_device_trace_buffer_t` and can use `IREE_AMDGPU_TRACE_BUFFER_SCOPE(trace_buffer);` to make the trace buffer active for the current function scope. The host-side tracing uses TLS to accomplish this instead. Once a trace buffer scope is activated within a function the `IREE_AMDGPU_TRACE_*` macros can be used to define instrumented zones, attach payloads like integers or strings, log messages, and plot values.

Which tracing features are available is determined by the `IREE_HAL_AMDGPU_TRACING_FEATURES` bitfield and it's possible to significantly reduce code size by disabling ones not required in a deployment. Since the tracing feature pulls in a non-trivial amount of code and embeds a lot of additional strings it is recommended to disable tracing entirely unless needed and otherwise only enable the features required. For example, a release deployment would have tracing disabled on both host and device while a tracing deployment may have tracing enabled but debug messages, instrumentation, and allocation tracking disabled to allow users to still get execution timings but not add the additional overhead of the other features. The `IREE_HAL_AMDGPU_HAS_TRACING_FEATURE` macro is used to guard any code relying on a particular feature.

During development `IREE_AMDGPU_DBG` is useful as a `printf` as it accepts a small but practical subset of `printf` format specifiers and routes the message into the host tracing tool. These are only enabled in release (`NDEBUG`) builds. Note that there's significant overhead involved in formatting strings on the device and even though it's only present in debug builds having any such logging checked in can severely degrade the usability of debug builds for any other purpose - to that end, `IREE_AMDGPU_DBG` should be treated like a `printf` and never checked in unless on cold paths or critical to daily development.

Tracing is implemented by a ringbuffer shared with the host that is populated by the device and flushed occasionally. Timestamps are captued in the device ("agent" in HSA) domain and require later translation into system times that correlate with other tools or timing sources.
