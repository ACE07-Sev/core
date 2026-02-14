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
    QCOIDOpTest, QCOTest,
    testing::Values(QCOTestCase{"Identity", identity, emptyQCO},
                    QCOTestCase{"SingleControlledIdentity",
                                singleControlledIdentity, emptyQCO},
                    QCOTestCase{"MultipleControlledIdentity",
                                multipleControlledIdentity, emptyQCO},
                    QCOTestCase{"NestedControlledIdentity",
                                nestedControlledIdentity, emptyQCO},
                    QCOTestCase{"TrivialControlledIdentity",
                                trivialControlledIdentity, emptyQCO},
                    QCOTestCase{"InverseIdentity", inverseIdentity, emptyQCO},
                    QCOTestCase{"InverseMultipleControlledIdentity",
                                inverseMultipleControlledIdentity, emptyQCO}),
    printTestName);

TEST_F(QCOTest, IdOpMatrix) {
  auto moduleOp = QCOProgramBuilder::build(context.get(), identity);
  ASSERT_TRUE(moduleOp);

  // Get the operation from the module
  auto funcOp = *moduleOp->getBody()->getOps<mlir::func::FuncOp>().begin();
  auto idOp = *funcOp.getBody().getOps<IdOp>().begin();
  const auto matrix = idOp.getUnitaryMatrix();

  // Get the definition of the matrix from the DD library
  const auto definition = dd::opToSingleQubitGateMatrix(qc::OpType::I);

  // Convert it to an Eigen matrix
  Eigen::Matrix2cd eigenDefinition;
  eigenDefinition << definition[0], definition[1], definition[2], definition[3];

  // Check if the matrices are equal
  ASSERT_TRUE(matrix.isApprox(eigenDefinition));
}
