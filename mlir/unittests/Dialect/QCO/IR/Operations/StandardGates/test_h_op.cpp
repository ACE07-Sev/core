/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/GateMatrixDefinitions.hpp"
#include "ir/operations/OpType.hpp"
#include "qco_programs.h"
#include "test_qco_ir.h"

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOHOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"H", h, h},
        QCOTestCase{"SingleControlledH", singleControlledH, singleControlledH},
        QCOTestCase{"MultipleControlledH", multipleControlledH,
                    multipleControlledH},
        QCOTestCase{"NestedControlledH", nestedControlledH,
                    multipleControlledH},
        QCOTestCase{"TrivialControlledH", trivialControlledH, h},
        QCOTestCase{"InverseH", inverseH, h},
        QCOTestCase{"InverseMultipleControlledH", inverseMultipleControlledH,
                    multipleControlledH}),
    printTestName);

TEST_F(QCOTest, HOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), h);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto hOp = *funcOp.getBody().getOps<HOp>().begin();
  const auto matrix = hOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::H);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
