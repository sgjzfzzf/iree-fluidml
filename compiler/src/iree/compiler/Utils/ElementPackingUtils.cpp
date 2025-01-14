// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/ElementPackingUtils.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler {

bool needToPackSubByteElementBitWidth(unsigned bitWidth) {
  // Require the original bit width to be some power of two for now to avoid
  // trickiness and weirdness of packing and cross-byte access.
  // Also disallow boolean values for now--they may require separate interface
  // choices.
  return bitWidth < 8 && llvm::isPowerOf2_32(bitWidth) && bitWidth != 1;
}

bool needToPackSubByteElements(RankedTensorType shapedType) {
  unsigned bitWidth = IREE::Util::getTypeBitWidth(shapedType.getElementType());
  // i1 with packed memory layout does not need to be extended.
  if (bitWidth == 1 && IREE::Encoding::hasPackedStorageAttr(shapedType)) {
    return true;
  }
  return needToPackSubByteElementBitWidth(bitWidth);
}

static Type legalizeStorageElementTypeImpl(Type elementType,
                                           bool isPackedStorage) {
  // Only handle integers; floats in MLIR all have aligned widths (today).
  auto intType = dyn_cast<IntegerType>(elementType);
  if (!intType)
    return elementType;

  unsigned bitWidth = intType.getWidth();
  if (bitWidth == 1 && isPackedStorage) {
    return elementType;
  }

  // For sub-byte elements, default to pack them into bytes.
  if (needToPackSubByteElementBitWidth(bitWidth))
    return elementType;

  // Otherwise, extend them to the next power-of-two bit width.
  unsigned alignedBitWidth =
      IREE::Util::getRoundedElementByteWidth(intType) * 8;
  if (alignedBitWidth == bitWidth)
    return elementType;
  return IntegerType::get(elementType.getContext(), alignedBitWidth,
                          intType.getSignedness());
}

Type legalizeTensorStorageElementType(Type type) {
  auto tensorType = llvm::cast<TensorType>(type);
  return legalizeStorageElementTypeImpl(
      tensorType.getElementType(), IREE::Encoding::hasPackedStorageAttr(type));
}

Value calculateStorageElementCountInBytes(Location loc,
                                          RankedTensorType shapedType,
                                          ValueRange dynamicDims,
                                          OpBuilder &builder) {
  Attribute encoding = shapedType.getEncoding();
  if (auto encodingLayoutAttr =
          dyn_cast_or_null<IREE::Encoding::EncodingLayoutAttrInterface>(
              encoding)) {
    return encodingLayoutAttr.calculateStorageSizeInBytes(
        loc, builder, shapedType, dynamicDims);
  }

  Type alignedElementType = legalizeTensorStorageElementType(shapedType);
  unsigned elementBits = IREE::Util::getTypeBitWidth(alignedElementType);

  bool isPackedStorage = IREE::Encoding::hasPackedStorageAttr(shapedType);
  bool isI1WithPackedStorage = elementBits == 1 && isPackedStorage;

  // Calculate all static dims first, if any.
  int64_t staticCount = 1;
  if (!isI1WithPackedStorage &&
      !needToPackSubByteElementBitWidth(elementBits)) {
    staticCount *= IREE::Util::getRoundedElementByteWidth(alignedElementType);
  }

  for (unsigned i = 0; i < shapedType.getRank(); ++i) {
    if (!shapedType.isDynamicDim(i))
      staticCount *= shapedType.getDimSize(i);
  }
  // Scale by dynamic dims, if present.
  auto value =
      builder.create<arith::ConstantIndexOp>(loc, staticCount).getResult();
  for (auto dim : dynamicDims) {
    value = builder.createOrFold<arith::MulIOp>(loc, value, dim);
  }
  // Sub-byte packing requires putting multiple elements in the same byte.
  if (isI1WithPackedStorage || needToPackSubByteElementBitWidth(elementBits)) {
    assert(8 % elementBits == 0);
    unsigned byteElements = 8 / elementBits;
    // TODO(antiagainst): We may want to emit runtime check to make sure this is
    // divisible.
    auto divisor = builder.create<arith::ConstantIndexOp>(loc, byteElements);
    if (!isPackedStorage && dynamicDims.empty() &&
        (staticCount * elementBits) % 8 != 0) {
      return nullptr;
    }
    value = builder.createOrFold<arith::CeilDivUIOp>(loc, value, divisor);
  }

  return value;
}

Value calculateStorageElementOffsetInBytes(Location loc,
                                           RankedTensorType originalType,
                                           Value linearizedIndex,
                                           OpBuilder &builder) {
  Type alignedElementType = legalizeTensorStorageElementType(originalType);
  unsigned elementBits = IREE::Util::getTypeBitWidth(alignedElementType);

  bool isPackedStorage = IREE::Encoding::hasPackedStorageAttr(originalType);
  bool isI1WithPackedStorage = elementBits == 1 && isPackedStorage;

  // Sub-byte packing requires putting multiple elements in the same byte.
  if (isI1WithPackedStorage || needToPackSubByteElementBitWidth(elementBits)) {
    Value byteElements =
        builder.create<arith::ConstantIndexOp>(loc, 8 / elementBits);
    // TODO(antiagainst): We may want to emit runtime check to make sure this is
    // divisible.
    return builder.createOrFold<arith::DivUIOp>(loc, linearizedIndex,
                                                byteElements);
  }

  Value elementBytes = builder.create<arith::ConstantIndexOp>(
      loc, IREE::Util::getRoundedElementByteWidth(alignedElementType));
  return builder.createOrFold<arith::MulIOp>(loc, linearizedIndex,
                                             elementBytes);
}

} // namespace mlir::iree_compiler
