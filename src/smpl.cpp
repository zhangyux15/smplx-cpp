#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include "smpl.h"
#include "math_util.h"


Smpl::Smpl(const SmplType& _type, const std::string &modelPath)
{
	m_type = _type;

	// vertices
	m_vertices = MathUtil::LoadMat<float>(modelPath + "/vertices.txt").transpose();
	m_vertexSize = int(m_vertices.cols());
	
	// faces
	m_faces = MathUtil::LoadMat<int>(modelPath + "/faces.txt").transpose();
	m_faceSize = int(m_faces.cols());

	// jregressor 
	m_jRegressor = MathUtil::LoadMat<float>(modelPath + "/jregressor.txt").transpose();
	assert(m_jRegressor.rows() == m_vertexSize && m_jRegressor.cols() == GetSmplDef(m_type).jointSize);

	// lbsweight
	m_lbsWeights = MathUtil::LoadMat<float>(modelPath + "/lbs_weights.txt").transpose();
	assert(m_lbsWeights.rows() == GetSmplDef(m_type).jointSize && m_lbsWeights.cols() == m_vertexSize);

	// vshapeblend
	m_vShapeBlend = MathUtil::LoadMat<float>(modelPath + "/shape_blend.txt");
	assert(m_vShapeBlend.rows() == m_vertexSize * 3 && m_vShapeBlend.cols() == GetSmplDef(m_type).shapeSize);

	// hand
	if (GetSmplDef(m_type).handJointSize > 0) {
		m_lhandMean = MathUtil::LoadMat<float>(modelPath + "/lhand_mean.txt");
		m_lhandComponents = MathUtil::LoadMat<float>(modelPath + "/lhand_components.txt").transpose();
		m_rhandMean = MathUtil::LoadMat<float>(modelPath + "/rhand_mean.txt");
		m_rhandComponents = MathUtil::LoadMat<float>(modelPath + "/rhand_components.txt").transpose();

		assert(m_lhandComponents.cols() == GetSmplDef(m_type).handPCASize && m_rhandComponents.cols() == GetSmplDef(m_type).handPCASize);
	}

	m_joints = m_vertices * m_jRegressor;
	m_jShapeBlend.resize(3 * GetSmplDef(m_type).jointSize, GetSmplDef(m_type).shapeSize);
	for (int shapeIdx = 0; shapeIdx < m_jShapeBlend.cols(); shapeIdx++) 
		Eigen::Map<Eigen::MatrixXf>(m_jShapeBlend.col(shapeIdx).data(), 3, GetSmplDef(m_type).jointSize)
			= Eigen::Map<Eigen::MatrixXf>(m_vShapeBlend.col(shapeIdx).data(), 3, m_vertexSize) * m_jRegressor;
}



Eigen::Matrix3Xf Smpl::CalcJBlend(const SmplParam& param) const
{
	const Eigen::VectorXf jOffset = m_jShapeBlend * param.GetShape();
	const Eigen::Matrix3Xf jBlend = m_joints + Eigen::Map<const Eigen::Matrix3Xf>(jOffset.data(), 3, m_joints.cols());
	return jBlend;
}


Eigen::VectorXf Smpl::CalcFullPose(const SmplParam& param) const
{
	Eigen::VectorXf fullPose(3 + 3 * GetSmplDef(m_type).jointSize);
	fullPose.head(3 + 3 * GetSmplDef(m_type).bodyJointSize) = param.GetTransPose();
	if (GetSmplDef(m_type).handJointSize > 0) {
		fullPose.segment(3 + 3 * GetSmplDef(m_type).bodyJointSize, 3 * GetSmplDef(m_type).handJointSize) = m_lhandMean + m_lhandComponents * param.GetLhandPCA();
		fullPose.tail(3 * GetSmplDef(m_type).handJointSize) = m_rhandMean + m_rhandComponents * param.GetRhandPCA();
	}
	return fullPose;
}


Eigen::Matrix4Xf Smpl::CalcNodeWarps(const Eigen::VectorXf& fullPose, const Eigen::Matrix3Xf& jBlend) const
{
	Eigen::Matrix4Xf nodeWarps = Eigen::Matrix4Xf(4, 4 * GetSmplDef(m_type).jointSize);
	for (int jIdx = 0; jIdx < GetSmplDef(m_type).jointSize; jIdx++) {
		Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
		if (GetSmplDef(m_type).parent[jIdx] == -1)
			matrix.topRightCorner(3, 1) = jBlend.col(jIdx) + fullPose.head(3);
		else
			matrix.topRightCorner(3, 1) = jBlend.col(jIdx) - jBlend.col(GetSmplDef(m_type).parent[jIdx]);

		matrix.topLeftCorner(3, 3) = MathUtil::Rodrigues<float>(fullPose.segment<3>(3 + 3 * jIdx));
		nodeWarps.block<4, 4>(0, 4 * jIdx) = matrix;
	}
	return nodeWarps;
}


Eigen::Matrix4Xf Smpl::CalcChainWarps(const Eigen::Matrix4Xf& nodeWarps) const
{
	Eigen::Matrix4Xf chainWarps(4, 4 * GetSmplDef(m_type).jointSize);
	for (int jIdx = 0; jIdx < GetSmplDef(m_type).jointSize; jIdx++) 
		if (jIdx == 0)
			chainWarps.middleCols(jIdx * 4, 4) = nodeWarps.middleCols(jIdx * 4, 4);
		else 
			chainWarps.middleCols(jIdx * 4, 4) = chainWarps.middleCols(GetSmplDef(m_type).parent[jIdx] * 4, 4) * nodeWarps.middleCols(jIdx * 4, 4);
	return chainWarps;
}


Eigen::Matrix3Xf Smpl::CalcJFinal(const Eigen::Matrix4Xf& chainWarps) const
{
	Eigen::Matrix3Xf jFinal(3, GetSmplDef(m_type).jointSize);
	for (int jIdx = 0; jIdx < jFinal.cols(); jIdx++)
		jFinal.col(jIdx) = chainWarps.block<3, 1>(0, 4 * jIdx + 3);
	return jFinal;
}


Eigen::Matrix3Xf Smpl::CalcJFinal(const SmplParam& param) const
{
	return CalcJFinal(CalcChainWarps(CalcNodeWarps(CalcFullPose(param), CalcJBlend(param))));
}


Eigen::Matrix3Xf Smpl::CalcVBlend(const SmplParam& param) const
{
	const Eigen::VectorXf vShapeOffset = m_vShapeBlend * param.GetShape();
	const Eigen::Matrix3Xf vBlend = m_vertices + Eigen::Map<const Eigen::MatrixXf>(vShapeOffset.data(), 3, m_vertexSize);
	return vBlend;
}


Eigen::Matrix3Xf Smpl::CalcVFinal(const Eigen::Matrix3Xf& jBlend, const Eigen::Matrix3Xf& vBlend, const Eigen::Matrix4Xf& chainWarps) const
{
	Eigen::Matrix4Xf chainWarpsNormalized = chainWarps;
	for (int jIdx = 0; jIdx < GetSmplDef(m_type).jointSize; jIdx++)
		chainWarpsNormalized.block<3, 1>(0, jIdx * 4 + 3) -= chainWarpsNormalized.block<3, 3>(0, jIdx * 4)*jBlend.col(jIdx);
	Eigen::Matrix3Xf vFinal(3, m_vertexSize);

#pragma omp parallel for
	for (int vIdx = 0; vIdx < m_vertexSize; vIdx++) {
		Eigen::Matrix4f warp;
		Eigen::Map<Eigen::VectorXf>(warp.data(), 16)
			= Eigen::Map<Eigen::MatrixXf>(chainWarpsNormalized.data(), 16, GetSmplDef(m_type).jointSize) * (m_lbsWeights.col(vIdx));
		vFinal.col(vIdx) = warp.topLeftCorner(3, 4)*(vBlend.col(vIdx).homogeneous());
	}
	return vFinal;
}


Eigen::Matrix3Xf Smpl::CalcVFinal(const SmplParam& param) const
{
	const Eigen::Matrix3Xf jBlend = CalcJBlend(param);
	const Eigen::Matrix3Xf vBlend = CalcVBlend(param);
	return CalcVFinal(jBlend, vBlend, CalcChainWarps(CalcNodeWarps(CalcFullPose(param), jBlend)));
}


void Smpl::SaveObj(const SmplParam& param, const std::string& filename) const
{
	const Eigen::Matrix3Xf vFinal = CalcVFinal(param);

	std::ofstream fs(filename);
	for (int i = 0; i < vFinal.cols(); i++)
		fs << "v " << vFinal(0, i) << " " << vFinal(1, i) << " " << vFinal(2, i) << std::endl;

	for (int i = 0; i < m_faces.cols(); i++)
		fs << "f " << m_faces(0, i) + 1 << " " << m_faces(1, i) + 1 << " " << m_faces(2, i) + 1 << std::endl;

	fs.close();
}


//// vposeblend (load in transpose)
//ifs.open(modelPath + "/pose_blend.txt");
//if (!ifs.is_open()) {
//	std::cerr << "cannot open " << modelPath + "/pose_blend.txt" << std::endl;
//	std::abort();
//}
//m_vPoseblend.resize(conf->vertexSize * 3, 9 * (conf->jointSize - 1));
//for (int col = 0; col < m_vPoseblend.cols(); col++)
//	for (int row = 0; row < m_vPoseblend.rows(); row++)
//		ifs >> m_vPoseblend(row, col);
//ifs.close();
//Eigen::Matrix3Xf poseFeature(3, 3 * (conf->jointSize - 1));
	//for (int jIdx = 1; jIdx < conf->jointSize; jIdx++)
	//	poseFeature.block<3, 3>(0, (jIdx - 1) * 3) = MathUtil::Rodrigues(param.GetPose(jIdx)).transpose() - Eigen::Matrix3f::Identity();
	//Eigen::VectorXf vPoseOffset = m_vPoseblend * Eigen::Map<Eigen::VectorXf>(poseFeature.data(), 9 * (conf->jointSize - 1));
	//
	//*vBlend = m_vertices + Eigen::Map<Eigen::MatrixXf>(vShapeOffset.data(), 3, conf->vertexSize) + Eigen::Map<Eigen::MatrixXf>(vPoseOffset.data(), 3, conf->vertexSize);

