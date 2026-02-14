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
    QCOZOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Z", MQT_NAMED_BUILDER(z), MQT_NAMED_BUILDER(z)},
        QCOTestCase{"SingleControlledZ", MQT_NAMED_BUILDER(singleControlledZ),
                    MQT_NAMED_BUILDER(singleControlledZ)},
        QCOTestCase{"MultipleControlledZ",
                    MQT_NAMED_BUILDER(multipleControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)},
        QCOTestCase{"NestedControlledZ", MQT_NAMED_BUILDER(nestedControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)},
        QCOTestCase{"TrivialControlledZ", MQT_NAMED_BUILDER(trivialControlledZ),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"InverseZ", MQT_NAMED_BUILDER(inverseZ),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"InverseMultipleControlledZ",
                    MQT_NAMED_BUILDER(inverseMultipleControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)}));

TEST_F(QCOTest, ZOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), z);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto zOp = *funcOp.getBody().getOps<ZOp>().begin();
  const auto matrix = zOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::Z);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
