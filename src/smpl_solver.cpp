#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include "smpl_solver.h"


SmplSolver::Status SmplSolver::InitStatus(const std::vector<std::shared_ptr<BasicTerm>>& terms,
	const bool& shapeValid, const bool& poseValid) const
{
	const SmplDef& def = GetSmplDef(m_type);
	Status status;

	// config mask
	status.validVMask.setZero(m_vertexSize);
	for (const auto& term : terms)
		term->ConfigMask(status.validVMask);
	status.validVertex = status.validVMask.array().count() > 0;

	// init ATA and ATb
	if (shapeValid) {
		status.jShapeJacobi.resize(3 * def.jointSize, def.shapeSize);
		if (status.validVertex) 
			status.vShapeJacobi.resize(3 * m_vertexSize, def.shapeSize);
		status.shapeATA.resize(def.shapeSize, def.shapeSize);
		status.shapeATb.resize(def.shapeSize);
	}

	if (poseValid) {
		const int poseATASize = 3 + 3 * def.bodyJointSize + 2 * def.handPCASize;

		status.jPoseJacobi.resize(3 * def.jointSize, poseATASize);
		status.jFullPoseJacobi.resize(3 * def.jointSize, 3 + 3 * def.jointSize);
		status.nodeWarpsJacobi.resize(9, 3 * def.jointSize);
		if (status.validVertex) {
			status.vPoseJacobi.resize(3 * m_vertexSize, poseATASize);
			status.vFullPoseJacobi.resize(3 * m_vertexSize, 3 + 3 * def.jointSize);
		}
		status.poseATA.resize(poseATASize, poseATASize);
		status.poseATb.resize(poseATASize);
	}
	return status;
}


void SmplSolver::CalcStatus(const SmplParam& param, Status& status, const int& _jCut) const
{
	const SmplDef& def = GetSmplDef(m_type);
	status.jCut = _jCut;
	status.param = param;
	status.jBlend = CalcJBlend(param);
	status.fullPose = CalcFullPose(param);
	status.nodeWarps = CalcNodeWarps(status.fullPose, status.jBlend);
	status.chainWarps = CalcChainWarps(status.nodeWarps);
	status.jFinal = CalcJFinal(status.chainWarps);
	if (status.validVertex) {
		status.vBlend = CalcVBlend(param);
		status.vFinal = CalcVFinal(status.jBlend, status.vBlend, status.chainWarps);
	}

	if (status.shapeATA.size() > 0) {
		status.shapeATA.setZero();
		status.shapeATb.setZero();

		// calculate joint jacobi
		for (int jIdx = 0; jIdx < status.jCut; jIdx++)
			if (jIdx == 0)
				status.jShapeJacobi.middleRows(3 * jIdx, 3) = m_jShapeBlend.middleRows(3 * jIdx, 3);
			else {
				const int prtIdx = def.parent[jIdx];
				status.jShapeJacobi.middleRows(3 * jIdx, 3) = status.jShapeJacobi.middleRows(3 * prtIdx, 3)
					+ status.chainWarps.block<3, 3>(0, 4 * prtIdx) * (m_jShapeBlend.middleRows(3 * jIdx, 3) - m_jShapeBlend.middleRows(3 * prtIdx, 3));
			}
		

		// calculate vertex jacobi
		if (status.validVertex) {
#pragma omp parallel for
			for (int vIdx = 0; vIdx < m_vertexSize; vIdx++) {
				if (!status.validVMask[vIdx])
					continue;
				for (int jIdx = 0; jIdx < status.jCut; jIdx++) {
					if (m_lbsWeights(jIdx, vIdx) < FLT_EPSILON)
						continue;
					status.vShapeJacobi.middleRows(3 * vIdx, 3) +=
						m_lbsWeights(jIdx, vIdx) * (status.jShapeJacobi.middleRows(3 * jIdx, 3) + status.chainWarps.block<3, 3>(0, 4 * jIdx)
							* (m_vShapeBlend.middleRows(3 * vIdx, 3) - m_jShapeBlend.middleRows(3 * jIdx, 3)));
				}
			}
		}
	}

	if (status.poseATA.size() > 0) {
		status.poseATA.setZero();
		status.poseATb.setZero();

		// calculate joint jacobi
		for (int jIdx = 0; jIdx < status.jCut; jIdx++)
			status.nodeWarpsJacobi.middleCols(3 * jIdx, 3) = MathUtil::RodriguesJacobi<float>(status.fullPose.segment<3>(3 + 3 * jIdx)).transpose();

		status.jFullPoseJacobi.setZero();
		for (int djIdx = 0; djIdx < status.jCut; djIdx++) {
			status.jFullPoseJacobi.block<3, 3>(3 * djIdx, 0).setIdentity();
			for (int dAxis = 0; dAxis < 3; dAxis++) {
				std::vector<Eigen::Matrix4f> dChainWarps(status.jCut, Eigen::Matrix4f::Zero());
				std::vector<int> valid(status.jCut, 0); valid[djIdx] = 1;
				dChainWarps[djIdx].topLeftCorner(3, 3) = Eigen::Map<Eigen::Matrix3f>(status.nodeWarpsJacobi.col(3 * djIdx + dAxis).data());
				if (djIdx != 0)
					dChainWarps[djIdx] = status.chainWarps.middleCols(4 * def.parent[djIdx], 4) * dChainWarps[djIdx];

				for (int jIdx = djIdx + 1; jIdx < status.jCut; jIdx++) {
					const int prtIdx = def.parent[jIdx];
					valid[jIdx] = valid[prtIdx];
					if (!valid[jIdx])
						continue;

					dChainWarps[jIdx] = dChainWarps[prtIdx] * status.nodeWarps.middleCols(4 * jIdx, 4);
					status.jFullPoseJacobi.block<3, 1>(jIdx * 3, 3 + djIdx * 3 + dAxis) = dChainWarps[jIdx].topRightCorner(3, 1);
				}
			}
		}

		status.jPoseJacobi.leftCols(3 + 3 * def.bodyJointSize) = status.jFullPoseJacobi.leftCols(3 + 3 * def.bodyJointSize);
		if (def.handPCASize > 0 && status.jCut >= def.bodyJointSize) {
			status.jPoseJacobi.middleCols(3 + 3 * def.bodyJointSize, def.handPCASize)
				= status.jFullPoseJacobi.middleCols(3 + 3 * def.bodyJointSize, 3 * def.handJointSize) * m_lhandComponents;
			status.jPoseJacobi.rightCols(def.handPCASize)
				= status.jFullPoseJacobi.rightCols(3 * def.handJointSize) * m_rhandComponents;
		}


		// calculate vertex jacobi
		if (status.validVertex) {
			status.vFullPoseJacobi.setZero();
#pragma omp parallel for
			for (int vIdx = 0; vIdx < m_vertexSize; vIdx++) {
				if (!status.validVMask[vIdx])
					continue;
				for (int jIdx = 0; jIdx < status.jCut; jIdx++) {
					if (m_lbsWeights(jIdx, vIdx) < FLT_EPSILON)
						continue;
					status.vFullPoseJacobi.middleRows(3 * vIdx, 3) += m_lbsWeights(jIdx, vIdx) * status.jFullPoseJacobi.middleRows(3 * jIdx, 3);
					const Eigen::Matrix3f skew = -MathUtil::Skew<float>(status.vFinal.col(vIdx) - status.jFinal.col(jIdx));
					for (int _prtIdx = jIdx; _prtIdx != -1; _prtIdx = def.parent[_prtIdx])
						status.vFullPoseJacobi.block<3, 3>(3 * vIdx, 3 + 3 * _prtIdx) += m_lbsWeights(jIdx, vIdx) * skew * status.chainWarps.block<3, 3>(0, 4 * _prtIdx);
				}
			}
			status.vPoseJacobi.leftCols(3 + 3 * def.bodyJointSize) = status.vFullPoseJacobi.leftCols(3 + 3 * def.bodyJointSize);
			if (def.handPCASize > 0 && status.jCut >= def.bodyJointSize) {
				status.vPoseJacobi.middleCols(3 + 3 * def.bodyJointSize, def.handPCASize)
					= status.vFullPoseJacobi.middleCols(3 + 3 * def.bodyJointSize, 3 * def.handJointSize) * m_lhandComponents;
				status.vPoseJacobi.rightCols(def.handPCASize)
					= status.vFullPoseJacobi.rightCols(3 * def.handJointSize) * m_rhandComponents;
			}
		}
	}
}


void SmplSolver::SolveShape(const std::vector<std::shared_ptr<BasicTerm>>& terms, SmplParam& param, const int& maxIterTime, const float& updateThresh) const
{
	const SmplDef& def = GetSmplDef(m_type);
	Status status = InitStatus(terms, true, false);

	for (int i = 0; i < maxIterTime; i++) {
		CalcStatus(param, status, def.jointSize);
		for (const auto& term : terms)
			term->CalcTerm(status);

		const Eigen::VectorXf delta = status.shapeATA.ldlt().solve(status.shapeATb);
		param.GetShape() += delta;

		if (delta.maxCoeff() < updateThresh)
			break;

		// debug
		// printf("iter: %d, update: %f\n", iterTime, deltaParam.data.norm());
	}
}


void SmplSolver::SolvePose(const std::vector<std::shared_ptr<BasicTerm>>& terms, SmplParam& param, const int& maxIterTime, const float& updateThresh) const
{
	const SmplDef& def = GetSmplDef(m_type);
	Status status = InitStatus(terms, false, true);

	for (int i = 0; i < maxIterTime; i++) {
		CalcStatus(param, status, def.jointSize);
		for (const auto& term : terms)
			term->CalcTerm(status);

		Eigen::VectorXf delta;
		if (status.jCut <= def.bodyJointSize) {
			delta = status.poseATA.topLeftCorner(
				3 + 3 * status.jCut, 3 + 3 * status.jCut).ldlt().solve(status.poseATb.head(3 + 3 * status.jCut));
			param.GetTransPose().head(3 + 3 * status.jCut) += delta;
		}
		else {
			delta = status.poseATA.ldlt().solve(status.poseATb);
			param.GetTransPosePCA().head(3 + 3 * status.jCut) += delta;
		}

		if (delta.maxCoeff() < updateThresh)
			break;

		// debug
		// printf("iter: %d, update: %f\n", iterTime, deltaParam.data.norm());
	}
}


void SmplSolver::SolvePoseHier(const std::vector<std::shared_ptr<BasicTerm>>& terms, SmplParam& param, const int& maxIterTime, const float& updateThresh) const
{
	const SmplDef& def = GetSmplDef(m_type);
	Status status = InitStatus(terms, false, true);

	for (int hierarchy = 0; hierarchy < def.hierarchyMap.size(); hierarchy++) {
		for (int i = 0; i < maxIterTime; i++) {
			CalcStatus(param, status, def.hierarchyMap[hierarchy]);
			for (const auto& term : terms)
				term->CalcTerm(status);

			Eigen::VectorXf delta;
			if (status.jCut <= def.bodyJointSize) {
				delta = status.poseATA.topLeftCorner(
					3 + 3 * status.jCut, 3 + 3 * status.jCut).ldlt().solve(status.poseATb.head(3 + 3 * status.jCut));
				param.GetTransPose().head(3 + 3 * status.jCut) += delta;
			}
			else {
				delta = status.poseATA.ldlt().solve(status.poseATb);
				param.GetTransPosePCA().head(3 + 3 * status.jCut) += delta;
			}

			if (delta.maxCoeff() < updateThresh)
				break;

			// debug
			// printf("iter: %d, update: %f\n", iterTime, deltaParam.data.norm());
		}
	}
}


void SmplSolver::AlignShape(const Eigen::Matrix3Xf& jTarget, SmplParam& param, 
	std::shared_ptr<SmplSolver::PriorTerm> priorTerm, std::shared_ptr<SmplSolver::RegularTerm> regularTerm, 
	const int& iterTime, const float& updateThresh) const
{
	const SmplDef& def = GetSmplDef(m_type);
	Eigen::Matrix3Xf jTargetTPose(3, jTarget.cols());
	for (int jIdx = 0; jIdx < jTarget.cols(); jIdx++) {
		if (jIdx == 0)
			jTargetTPose.col(jIdx) = m_joints.col(jIdx);
		else {
			const int prtIdx = def.parent[jIdx];
			jTargetTPose.col(jIdx) = (jTarget.col(jIdx) - jTarget.col(prtIdx)).norm()
				* (m_joints.col(jIdx) - m_joints.col(prtIdx)).normalized() + jTargetTPose.col(prtIdx);
		}
	}

	SmplParam _param(m_type);
	std::vector<std::shared_ptr<BasicTerm>> terms;
	terms.emplace_back(std::make_shared<J3dTerm>(jTargetTPose.colwise().homogeneous()));
	terms.emplace_back(priorTerm);
	terms.emplace_back(regularTerm);
	SolveShape(terms, _param, iterTime, updateThresh);
	param.GetShape() = _param.GetShape();
}


void SmplSolver::AlignRT(const Eigen::Matrix3Xf& jTarget, SmplParam& param) const
{
	// align root affine
	param.GetTrans() = jTarget.col(0) - m_joints.col(0);
	auto CalcAxes = [](const Eigen::Vector3f& xAxis, const Eigen::Vector3f& yAxis) {
		Eigen::Matrix3f axes;
		axes.col(0) = xAxis.normalized();
		axes.col(2) = xAxis.cross(yAxis).normalized();
		axes.col(1) = axes.col(2).cross(axes.col(0)).normalized();
		return axes;
	};
	const Eigen::AngleAxisf angleAxis(CalcAxes(jTarget.col(2) - jTarget.col(1), jTarget.col(3) - jTarget.col(1))
		* (CalcAxes(m_joints.col(2) - m_joints.col(1), m_joints.col(3) - m_joints.col(1)).inverse()));
	param.GetPose().head(3) = angleAxis.axis() * angleAxis.angle();
}


void SmplSolver::AlignPose(const Eigen::Matrix3Xf& jTarget, SmplParam& param, 
	std::shared_ptr<SmplSolver::PriorTerm> priorTerm, std::shared_ptr<SmplSolver::RegularTerm> regularTerm, 
	const int& iterTime, const float& updateThresh) const
{
	AlignRT(jTarget, param);
	std::vector<std::shared_ptr<BasicTerm>> terms;
	terms.emplace_back(std::make_shared<J3dTerm>(jTarget.colwise().homogeneous()));
	terms.emplace_back(priorTerm);
	terms.emplace_back(regularTerm);
	SolvePose(terms, param, iterTime, updateThresh);
}


void SmplSolver::J3dTerm::CalcTerm(Status& status) const
{
	const SmplDef& def = GetSmplDef(status.param.type);
	for (int jIdx = 0; jIdx < jTarget.cols() && jIdx < status.jCut; jIdx++) {
		if (jTarget(3, jIdx) < FLT_EPSILON)
			continue;

		const float ww = jTarget(3, jIdx);
		const Eigen::Vector3f residual = jTarget.block<3, 1>(0, jIdx) - status.jFinal.col(jIdx);
		if (status.shapeATA.size() > 0) {
			auto jacobi = status.jShapeJacobi.middleRows(3 * jIdx, 3);
			status.shapeATA += w * ww *  jacobi.transpose() * jacobi;
			status.shapeATb += w * ww * jacobi.transpose()*residual;
		}
		if (status.poseATA.size() > 0) {
			if (status.jCut <= def.bodyJointSize) {
				auto jacobi = status.jPoseJacobi.middleRows(3 * jIdx, 3).leftCols(3 + 3 * status.jCut);
				status.poseATA.topLeftCorner(3 + 3 * status.jCut, 3 + 3 * status.jCut) += w * ww *  jacobi.transpose() * jacobi;
				status.poseATb.head(3 + 3 * status.jCut) += w * ww * jacobi.transpose()*residual;
			}
			else {
				auto jacobi = status.jPoseJacobi.middleRows(3 * jIdx, 3);
				status.poseATA += w * ww *  jacobi.transpose() * jacobi;
				status.poseATb += w * ww * jacobi.transpose()*residual;
			}
		}
	}
}


void SmplSolver::J2dTerm::CalcTerm(Status& status) const
{
	const SmplDef& def = GetSmplDef(status.param.type);
	for (int camIdx = 0; camIdx < projs.size(); camIdx++) {
		for (int jIdx = 0; jIdx < jTarget2d[camIdx].cols() && jIdx < status.jCut; jIdx++) {
			if (jTarget2d[camIdx](2, jIdx) < FLT_EPSILON)
				continue;

			const float ww = jTarget2d[camIdx](2, jIdx);
			const Eigen::Vector3f abc = projs[camIdx] * (status.jFinal.col(jIdx).homogeneous());
			Eigen::Matrix<float, 2, 3> projJacobi;
			projJacobi << 1.0f / abc.z(), 0.0f, -abc.x() / (abc.z()*abc.z()),
				0.0f, 1.0f / abc.z(), -abc.y() / (abc.z()*abc.z());
			projJacobi = projJacobi * projs[camIdx].leftCols(3);
			const Eigen::Vector2f residual = jTarget2d[camIdx].block<2, 1>(0, jIdx) - abc.hnormalized();
			if (status.shapeATA.size() > 0) {
				auto jacobi = projJacobi * status.jShapeJacobi.middleRows(3 * jIdx, 3);
				status.shapeATA += w * ww *  jacobi.transpose() * jacobi;
				status.shapeATb += w * ww * jacobi.transpose()*residual;
			}
			if (status.poseATA.size() > 0) {
				if (status.jCut <= def.bodyJointSize) {
					auto jacobi = projJacobi * (status.jPoseJacobi.middleRows(3 * jIdx, 3).leftCols(3 + 3 * status.jCut));
					status.poseATA.topLeftCorner(3 + 3 * status.jCut, 3 + 3 * status.jCut) += w * ww *  jacobi.transpose() * jacobi;
					status.poseATb.head(3 + 3 * status.jCut) += w * ww * jacobi.transpose()*residual;
				}
				else {
					auto jacobi = projJacobi * status.jPoseJacobi.middleRows(3 * jIdx, 3);
					status.poseATA += w * ww *  jacobi.transpose() * jacobi;
					status.poseATb += w * ww * jacobi.transpose()*residual;
				}
			}
		}
	}
}


void SmplSolver::TemporalTerm::CalcTerm(Status& status) const
{
	const SmplDef& def = GetSmplDef(status.param.type);
	if (status.shapeATA.size() > 0) {
		status.shapeATA.topLeftCorner(def.somatotypeSize, def.somatotypeSize) 
			+= w * wSomatotype * Eigen::MatrixXf::Identity(def.somatotypeSize, def.somatotypeSize);
		status.shapeATb.head(def.somatotypeSize) += w * wSomatotype * (paramPrev.GetSomatotype() - status.param.GetSomatotype());
		if (def.expressionSize > 0) {
			status.shapeATA.bottomRightCorner(def.expressionSize, def.expressionSize)
				+= w * wExpression * Eigen::MatrixXf::Identity(def.expressionSize, def.expressionSize);
			status.shapeATb.tail(def.expressionSize) += w * wExpression * (paramPrev.GetExpression() - status.param.GetExpression());
		}
	}
	if (status.poseATA.size() > 0) {
		status.poseATA.topLeftCorner(3, 3) += w * wTrans * Eigen::MatrixXf::Identity(3, 3);
		status.poseATb.head(3) += w * wTrans * (paramPrev.GetTrans() - status.param.GetTrans());

		if (status.jCut <= def.bodyJointSize) {
			status.poseATA.block(3, 3, 3 * status.jCut, 3 * status.jCut)
				+= w * wPoseBody * Eigen::MatrixXf::Identity(3 * status.jCut, 3 * status.jCut);
			status.poseATb.segment(3, 3 * status.jCut) += w * wPoseBody * (
				paramPrev.GetPose().head(3 * status.jCut) - status.param.GetPose().head(3 * status.jCut));
		}
		else {
			status.poseATA.block(3, 3, 3 * def.bodyJointSize, 3 * def.bodyJointSize)
				+= w * wPoseBody * Eigen::MatrixXf::Identity(3 * def.bodyJointSize, 3 * def.bodyJointSize);
			status.poseATb.segment(3, 3 * def.bodyJointSize) += w * wPoseBody * (paramPrev.GetPose() - status.param.GetPose());
			if (def.handPCASize > 0) {
				status.poseATA.bottomRightCorner(2 * def.handPCASize, 2 * def.handPCASize)
					+= w * wPoseHand * Eigen::MatrixXf::Identity(2 * def.handPCASize, 2 * def.handPCASize);
				status.poseATb.tail(2 * def.handPCASize) += w * wPoseHand * (paramPrev.GetHandPCA() - status.param.GetHandPCA());
			}
		}
	}
}


void SmplSolver::V3dTerm::ConfigMask(Eigen::VectorXf& validVMask) const
{
	for (int vIdx = 0; vIdx < validVMask.size(); vIdx++)
		if (!validVMask[vIdx] && vIdx < vTarget.cols() && vTarget(3, vIdx) >= FLT_EPSILON)
			validVMask[vIdx] = 1;
}


void SmplSolver::V3dTerm::CalcTerm(Status& status) const
{
	const SmplDef& def = GetSmplDef(status.param.type);
	if (status.jCut != def.jointSize)
		return;

	for (int vIdx = 0; vIdx < vTarget.cols(); vIdx++) {
		if (!status.validVMask[vIdx] || vTarget(3, vIdx) < FLT_EPSILON)
			continue;

		const float ww = vTarget(3, vIdx);
		const Eigen::Vector3f residual = vTarget.block<3, 1>(0, vIdx) - status.vFinal.col(vIdx);
		if (status.shapeATA.size() > 0) {
			auto jacobi = status.vShapeJacobi.middleRows(3 * vIdx, 3);
			status.shapeATA += w * ww *  jacobi.transpose() * jacobi;
			status.shapeATb += w * ww * jacobi.transpose()*residual;
		}
		if (status.poseATA.size() > 0) {
			auto jacobi = status.vPoseJacobi.middleRows(3 * vIdx, 3);
			status.poseATA += w * ww *  jacobi.transpose() * jacobi;
			status.poseATb += w * ww * jacobi.transpose()*residual;
		}
	}
}


void SmplSolver::V2dTerm::ConfigMask(Eigen::VectorXf& validVMask) const
{
	for (int camIdx = 0; camIdx < projs.size(); camIdx++)
		for (int vIdx = 0; vIdx < validVMask.size(); vIdx++)
			if (!validVMask[vIdx] && vIdx < vTarget2d[camIdx].cols() && vTarget2d[camIdx](2, vIdx) >= FLT_EPSILON)
				validVMask[vIdx] = 1;
}


void SmplSolver::V2dTerm::CalcTerm(Status& status) const
{
	const SmplDef& def = GetSmplDef(status.param.type);
	if (status.jCut != def.jointSize)
		return;

	for (int camIdx = 0; camIdx < projs.size(); camIdx++) {
		for (int vIdx = 0; vIdx < vTarget2d[camIdx].cols(); vIdx++) {
			if (!status.validVMask[vIdx] || vTarget2d[camIdx](2, vIdx) < FLT_EPSILON)
				continue;

			const float ww = vTarget2d[camIdx](2, vIdx);
			const Eigen::Vector3f abc = projs[camIdx] * (status.vFinal.col(vIdx).homogeneous());
			Eigen::Matrix<float, 2, 3> projJacobi;
			projJacobi << 1.0f / abc.z(), 0.0f, -abc.x() / (abc.z()*abc.z()),
				0.0f, 1.0f / abc.z(), -abc.y() / (abc.z()*abc.z());
			projJacobi = projJacobi * projs[camIdx].leftCols(3);
			const Eigen::Vector2f residual = vTarget2d[camIdx].block<2, 1>(0, vIdx) - abc.hnormalized();
			if (status.shapeATA.size() > 0) {
				const Eigen::Matrix2Xf jacobi = projJacobi * status.vShapeJacobi.middleRows(3 * vIdx, 3);
				status.shapeATA += w * ww *  jacobi.transpose() * jacobi;
				status.shapeATb += w * ww * jacobi.transpose()*residual;
			}
			if (status.poseATA.size() > 0) {
				const Eigen::Matrix2Xf jacobi = projJacobi * status.vPoseJacobi.middleRows(3 * vIdx, 3);
				status.poseATA += w * ww *  jacobi.transpose() * jacobi;
				status.poseATb += w * ww * jacobi.transpose()*residual;
			}
		}
	}
}


void SmplSolver::PriorTerm::CalcTerm(Status& status) const
{
	const SmplType type = status.param.type;
	const SmplDef& def = GetSmplDef(type);

	if (status.shapeATA.size() > 0) {
		switch (type)
		{
		case SMPLX:	
			status.shapeATA.topLeftCorner(def.somatotypeSize, def.somatotypeSize)
				+= w * wSomatotype * Eigen::MatrixXf::Identity(def.somatotypeSize, def.somatotypeSize);
			status.shapeATb.head(def.somatotypeSize) -= w * wSomatotype * status.param.GetSomatotype();

			status.shapeATA.bottomRightCorner(def.expressionSize, def.expressionSize)
				+= w * wExpression * Eigen::MatrixXf::Identity(def.expressionSize, def.expressionSize);
			status.shapeATb.tail(def.expressionSize) -= w * wExpression * status.param.GetExpression();
			break;

		case SMPLH: case SMPL:	
			status.shapeATA += w * wSomatotype * Eigen::MatrixXf::Identity(status.shapeATA.rows(), status.shapeATA.rows());
			status.shapeATb -= w * wSomatotype * status.param.GetShape();
			break;
		default:
			std::cerr << "unknow smpl type" << std::endl;
			std::abort();
			break;
		}
	}

	if (status.poseATA.size() > 0 && status.jCut == def.jointSize) {
		Eigen::MatrixXf bodyATA, lhandATA, rhandATA;
		Eigen::VectorXf bodyATb, lhandATb, rhandATb;

		switch (type)
		{
		case SMPLX:
			// body
			bodyGmm.CalcTerm(status.param.GetTransPose().segment(6, 63), bodyATA, bodyATb);
			status.poseATA.block(6, 6, 63, 63) += w * wPoseBody * bodyATA;
			status.poseATb.segment(6, 63) += w * wPoseBody * bodyATb;

			// head
			status.poseATA.block(69, 69, 9, 9) += w * wPoseBody * Eigen::MatrixXf::Identity(9, 9);
			status.poseATb.segment(69, 9) -= w * wPoseBody * status.param.GetTransPose().segment(69, 9);

			// hand
			status.poseATA.bottomRightCorner(2 * def.handPCASize, 2 * def.handPCASize)
				+= w * wPoseHand * Eigen::MatrixXf::Identity(2 * def.handPCASize, 2 * def.handPCASize);
			status.poseATb.tail(2 * def.handPCASize) -= w * wPoseHand * status.param.GetHandPCA();

			break;
		case SMPLH:
			// body
			bodyGmm.CalcTerm(status.param.GetTransPose().segment(6, 63), bodyATA, bodyATb);
			status.poseATA.block(6, 6, 63, 63) += w * wPoseBody * bodyATA;
			status.poseATb.segment(6, 63) += w * wPoseBody * bodyATb;

			// hand
			status.poseATA.bottomRightCorner(2 * def.handPCASize, 2 * def.handPCASize)
				+= w * wPoseHand * Eigen::MatrixXf::Identity(2 * def.handPCASize, 2 * def.handPCASize);
			status.poseATb.tail(2 * def.handPCASize) -= w * wPoseHand * status.param.GetHandPCA();

			break;
		case SMPL:
			// body
			bodyGmm.CalcTerm(status.param.GetPose().tail(69), bodyATA, bodyATb);
			status.poseATA.bottomRightCorner(69, 69) += w * wPoseBody * bodyATA;
			status.poseATb.tail(69) += w * wPoseBody * bodyATb;
			break;

		default:
			std::cerr << "unknow smpl type" << std::endl;
			std::abort();
			break;
		}
	}
}


void SmplSolver::RegularTerm::CalcTerm(Status& status) const
{
	const SmplType type = status.param.type;
	const SmplDef& def = GetSmplDef(type);
	if (status.shapeATA.size() > 0) {
		status.shapeATA.topLeftCorner(def.somatotypeSize, def.somatotypeSize)
			+= w * wSomatotype * Eigen::MatrixXf::Identity(def.somatotypeSize, def.somatotypeSize);
		if (def.expressionSize > 0)
			status.shapeATA.bottomRightCorner(def.expressionSize, def.expressionSize)
			+= w * wExpression * Eigen::MatrixXf::Identity(def.expressionSize, def.expressionSize);
	}
	if (status.poseATA.size() > 0) {
		if (status.jCut <= def.bodyJointSize) {
			status.poseATA.topLeftCorner(3 + 3 * status.jCut, 3 + 3 * status.jCut)
				+= w * wPoseBody * Eigen::MatrixXf::Identity(3 + 3 * status.jCut, 3 + 3 * status.jCut);
		}
		else {
			status.poseATA.topLeftCorner(3 + 3 * def.bodyJointSize, 3 + 3 * def.bodyJointSize)
				+= w * wPoseBody * Eigen::MatrixXf::Identity(3 + 3 * def.bodyJointSize, 3 + 3 * def.bodyJointSize);
			if (def.handPCASize > 0)
				status.poseATA.bottomRightCorner(2 * def.handPCASize, 2 * def.handPCASize)
				+= w * wPoseHand * Eigen::MatrixXf::Identity(2 * def.handPCASize, 2 * def.handPCASize);
		}
	}
}







//status.poseATA.block(6, 6, 3, 3) += w * wPoseBody * 1e-1f*Eigen::Matrix3f::Identity();
//status.poseATb.segment(6, 3) -= w * wPoseBody * 1e-1f*status.param.GetTransPose().segment(6, 3);
//status.poseATA.block(21, 21, 3, 3) += w * wPoseBody * 1e-1f*Eigen::Matrix3f::Identity();
//status.poseATb.segment(21, 3) -= w * wPoseBody * 1e-1f*status.param.GetTransPose().segment(21, 3);
//status.poseATA.block(30, 30, 3, 3) += w * wPoseBody * 1e-1f*Eigen::Matrix3f::Identity();
//status.poseATb.segment(30, 3) -= w * wPoseBody * 1e-1f*status.param.GetTransPose().segment(30, 3);
//status.poseATA.block(42, 42, 6, 6) += w * wPoseBody * 1e-1f*Eigen::MatrixXf::Identity(6, 6);
//status.poseATb.segment(42, 6) -= w * wPoseBody * 1e-1f*status.param.GetTransPose().segment(42, 6);


//
//void SmplSolver::AlignRT(const Eigen::Matrix3Xf& jTarget, SmplParam& param) const
//{
//	// align root affine
//	param.GetTrans() = jTarget.col(0) - m_joints.col(0);
//
//	auto CalcAxes = [](const Eigen::Vector3f& xAxis, const Eigen::Vector3f& yAxis) {
//		Eigen::Matrix3f axes;
//		axes.col(0) = xAxis.normalized();
//		axes.col(2) = xAxis.cross(yAxis).normalized();
//		axes.col(1) = axes.col(2).cross(axes.col(0)).normalized();
//		return axes;
//	};
//	param.GetPose().head(3) = MathUtil::InvRodrigues(CalcAxes(jTarget.col(2) - jTarget.col(1), jTarget.col(3) - jTarget.col(1))
//		* (CalcAxes(m_joints.col(2) - m_joints.col(1), m_joints.col(3) - m_joints.col(1)).inverse()));
//}
//

//status.poseATA.block<9, 9>(6, 6) += w * wPoseBody * Eigen::MatrixXf::Identity(9, 9);
//status.poseATb.segment<9>(6) -= w * wPoseBody * status.param.GetTransPose().segment<9>(6);

//status.poseATA.block<3, 3>(21, 21) += w * wPoseBody * Eigen::Matrix3f::Identity();
//status.poseATb.segment<3>(21) -= w * wPoseBody * status.param.GetTransPose().segment<3>(21);

//status.poseATA.block<3, 3>(30, 30) += w * wPoseBody * Eigen::Matrix3f::Identity();
//status.poseATb.segment<3>(30) -= w * wPoseBody * status.param.GetTransPose().segment<3>(30);

//status.poseATA.block<6, 6>(33, 33) += w * 10.f *  wPoseBody * Eigen::MatrixXf::Identity(6, 6);
//status.poseATb.segment<6>(33) -= w * 10.f * wPoseBody * status.param.GetTransPose().segment<6>(33);

//status.poseATA.block<12, 12>(39, 39) += w *  wPoseBody * Eigen::MatrixXf::Identity(12, 12);
//status.poseATb.segment<12>(39) -= w * wPoseBody * status.param.GetTransPose().segment<12>(39);
//
//status.poseATA.block<6, 6>(69, 69) += w * 0.1f* wPoseBody * Eigen::MatrixXf::Identity(6, 6);
//status.poseATb.segment<6>(69) -= w * 0.1f* wPoseBody * status.param.GetTransPose().segment<6>(69);