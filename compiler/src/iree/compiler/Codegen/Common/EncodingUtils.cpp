// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <numeric>

namespace mlir::iree_compiler {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Encoding::EncodingAttr;
using IREE::Encoding::getEncodingAttr;
using IREE::Encoding::getEncodingContractionDims;

MaterializeEncodingTypeConverter::MaterializeEncodingTypeConverter(
    IREE::Codegen::LayoutAttrInterface layoutAttr)
    : layoutAttr(layoutAttr) {
  addConversion([](IntegerType intType) { return intType; });
  addConversion([](IndexType indexType) { return indexType; });
  addConversion([](FloatType floatType) { return floatType; });
  addConversion([](MemRefType memrefType) { return memrefType; });
  addConversion([=](RankedTensorType type) -> RankedTensorType {
    MaterializeEncodingInfo encodingInfo = getEncodingInfo(type);
    if (IREE::Encoding::hasPackedStorageAttr(type)) {
      return type;
    }
    // For a given tensor type with an encoding, return the materialized
    // type to use for it. If no encoding is set, then return the tensor type
    // itself.
    if (IREE::Codegen::isIdentityLayout(encodingInfo)) {
      return dropEncoding(type);
    }
    auto packedType = cast<RankedTensorType>(tensor::PackOp::inferPackedType(
        type, encodingInfo.innerTileSizes, encodingInfo.innerDimsPos,
        encodingInfo.outerDimsPerm));

    // There is no swizzle, we are already done. Typically the case on CPU.
    if (!encodingInfo.swizzle) {
      return packedType;
    }

    // There is a swizzle, we need to handle it. Typically the case on GPU.
    auto swizzle = *encodingInfo.swizzle;
    SmallVector<int64_t> newShape(
        packedType.getShape().drop_back(encodingInfo.innerTileSizes.size()));
    SmallVector<int64_t> swizzledTileShape =
        IREE::Codegen::getExpandedTileShape(swizzle.expandShape);
    applyPermutationToVector(swizzledTileShape, swizzle.permutation);
    newShape.append(swizzledTileShape);
    return RankedTensorType::get(newShape, packedType.getElementType());
  });
}

MaterializeEncodingConversionTarget::MaterializeEncodingConversionTarget(
    MLIRContext &context)
    : ConversionTarget(context) {
  // Mark any operation that has operands/results with encoding as
  // illegal.
  markUnknownOpDynamicallyLegal([](Operation *op) {
    auto typeHasDataTilingEncoding = [](Type t) -> bool {
      auto tensorType = dyn_cast<RankedTensorType>(t);
      if (!tensorType)
        return false;
      return getEncodingAttr(tensorType) != nullptr;
    };
    auto valueHasDataTilingEncoding = [=](Value v) -> bool {
      return typeHasDataTilingEncoding(v.getType());
    };
    bool hasOperandOrResultsWithEncoding =
        llvm::any_of(op->getOperands(), valueHasDataTilingEncoding) ||
        llvm::any_of(op->getResultTypes(), typeHasDataTilingEncoding);
    return !hasOperandOrResultsWithEncoding;
  });
}

RankedTensorType dropEncoding(RankedTensorType type) {
  assert(!IREE::Encoding::hasPackedStorageAttr(type) &&
         "not expected `packed_storage` attribute.");
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

RankedTensorType dropPackedStorageEncodingIfAny(RankedTensorType type) {
  if (!IREE::Encoding::hasPackedStorageAttr(type))
    return type;
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

} // namespace mlir::iree_compiler
