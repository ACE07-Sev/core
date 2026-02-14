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
    QCOSXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SX", MQT_NAMED_BUILDER(sx), MQT_NAMED_BUILDER(sx)},
        QCOTestCase{"SingleControlledSX", MQT_NAMED_BUILDER(singleControlledSx),
                    MQT_NAMED_BUILDER(singleControlledSx)},
        QCOTestCase{"MultipleControlledSX",
                    MQT_NAMED_BUILDER(multipleControlledSx),
                    MQT_NAMED_BUILDER(multipleControlledSx)},
        QCOTestCase{"NestedControlledSX", MQT_NAMED_BUILDER(nestedControlledSx),
                    MQT_NAMED_BUILDER(multipleControlledSx)},
        QCOTestCase{"TrivialControlledSX",
                    MQT_NAMED_BUILDER(trivialControlledSx),
                    MQT_NAMED_BUILDER(sx)},
        QCOTestCase{"InverseSX", MQT_NAMED_BUILDER(inverseSx),
                    MQT_NAMED_BUILDER(sxdg)},
        QCOTestCase{"InverseMultipleControlledSX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledSx),
                    MQT_NAMED_BUILDER(multipleControlledSxdg)}));

TEST_F(QCOTest, SXOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), sx);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto sxOp = *funcOp.getBody().getOps<SXOp>().begin();
  const auto matrix = sxOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::SX);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
