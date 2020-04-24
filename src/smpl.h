#pragma once
#include <Eigen/Core>
#include <string>
#include <vector>
#include "math_util.h"


enum SmplType
{
	SMPL_TYPE_NONE = -1,
	SMPL,
	SMPLH,
	SMPLX,
	SMPL_TYPE_SIZE
};


struct SmplDef
{
	int bodyJointSize;
	int headJointSize = 0;		// include in bodyJointSize
	int handJointSize = 0;
	int jointSize;
	int somatotypeSize = 0;
	int expressionSize = 0;
	int shapeSize = 0;
	int handPCASize = 0;
	Eigen::VectorXi parent;
	Eigen::VectorXi hierarchyMap;
};


inline const SmplDef& GetSmplDef(const SmplType& type)
{
	static const std::vector<SmplDef> smplDefs = [] {
		std::vector<SmplDef> _smplDefs(SMPL_TYPE_SIZE);

		// SMPL24
		_smplDefs[SMPL].bodyJointSize = 24;
		_smplDefs[SMPL].jointSize = 24;
		_smplDefs[SMPL].somatotypeSize = 10;
		_smplDefs[SMPL].shapeSize = 10;
		_smplDefs[SMPL].parent.resize(24);
		_smplDefs[SMPL].parent << -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21;
		_smplDefs[SMPL].hierarchyMap.resize(24);
		_smplDefs[SMPL].hierarchyMap << 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7;

		// SMPLH
		_smplDefs[SMPLH].bodyJointSize = 22;
		_smplDefs[SMPLH].handJointSize = 15;
		_smplDefs[SMPLH].jointSize = 52;
		_smplDefs[SMPLH].somatotypeSize = 10;
		_smplDefs[SMPLH].shapeSize = 10;
		_smplDefs[SMPLH].handPCASize = 30;
		_smplDefs[SMPLH].parent.resize(52);
		_smplDefs[SMPLH].parent << -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,		// body
			20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,								// lhand
			21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50;								// rhand
		_smplDefs[SMPLH].hierarchyMap.resize(52);
		_smplDefs[SMPLH].hierarchyMap << 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6,
			7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
			7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7;

		// SMPLX
		_smplDefs[SMPLX].bodyJointSize = 25;
		_smplDefs[SMPLX].headJointSize = 3;
		_smplDefs[SMPLX].handJointSize = 15;
		_smplDefs[SMPLX].jointSize = 55;
		_smplDefs[SMPLX].somatotypeSize = 10;
		_smplDefs[SMPLX].expressionSize = 10;
		_smplDefs[SMPLX].shapeSize = 20;
		_smplDefs[SMPLX].handPCASize = 30;
		_smplDefs[SMPLX].parent.resize(55);
		_smplDefs[SMPLX].parent << -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,		// body
			15, 15, 15,																				// head
			20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,								// lhand
			21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53;								// rhand
		
		_smplDefs[SMPLX].hierarchyMap.resize(55);
		_smplDefs[SMPLX].hierarchyMap << 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6,
			6, 6, 6,
			7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
			7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7;

		return _smplDefs;
	}();

	assert(type >= 0 && type < SMPL_TYPE_SIZE);
	return smplDefs[type];
}


struct SmplParam
{
	SmplType type;
	Eigen::VectorXf data;

	SmplParam() { type = SMPL_TYPE_NONE; }
	SmplParam(const SmplType& _type) {
		type = _type;
		data.setZero(3 + 3 * GetSmplDef(type).bodyJointSize + 2* GetSmplDef(type).handPCASize + GetSmplDef(type).shapeSize);
	}

	auto GetTrans() { return data.segment<3>(0); }
	auto GetTrans() const { return data.segment<3>(0); }

	auto GetPose() { return data.segment(3, GetSmplDef(type).bodyJointSize * 3); }
	auto GetPose() const { return data.segment(3, GetSmplDef(type).bodyJointSize * 3); }

	auto GetTransPose() { return data.head(3 + GetSmplDef(type).bodyJointSize * 3); }
	auto GetTransPose() const { return data.head(3 + GetSmplDef(type).bodyJointSize * 3); }

	auto GetHandPCA() { return data.segment(3 + GetSmplDef(type).bodyJointSize * 3, 2 * GetSmplDef(type).handPCASize); }
	auto GetHandPCA() const { return data.segment(3 + GetSmplDef(type).bodyJointSize * 3, 2 * GetSmplDef(type).handPCASize); }

	auto GetLhandPCA() { return GetHandPCA().head(GetSmplDef(type).handPCASize); }
	auto GetLhandPCA() const { return GetHandPCA().head(GetSmplDef(type).handPCASize); }

	auto GetRhandPCA() { return GetHandPCA().tail(GetSmplDef(type).handPCASize); }
	auto GetRhandPCA() const { return GetHandPCA().tail(GetSmplDef(type).handPCASize); }

	auto GetTransPoseHand() { return data.head(3 + 3 * GetSmplDef(type).bodyJointSize + 2 * GetSmplDef(type).handPCASize); }
	auto GetTransPoseHand() const { return data.head(3 + 3 * GetSmplDef(type).bodyJointSize + 2 * GetSmplDef(type).handPCASize); }

	auto GetShape() { return data.tail(GetSmplDef(type).shapeSize); }
	auto GetShape() const { return data.tail(GetSmplDef(type).shapeSize); }

	auto GetSomatotype() { return GetShape().head(GetSmplDef(type).somatotypeSize); }
	auto GetSomatotype() const { return GetShape().head(GetSmplDef(type).somatotypeSize); }

	auto GetExpression() { return GetShape().tail(GetSmplDef(type).expressionSize); }
	auto GetExpression() const { return GetShape().tail(GetSmplDef(type).expressionSize); }
};


class Smpl
{
public:
	Smpl(const SmplType& _type, const std::string &modelPath);
	virtual ~Smpl() = default;

	Eigen::VectorXf CalcFullPose(const SmplParam& param, const int& _jCut = -1) const;
	Eigen::Matrix4Xf CalcNodeWarps(const Eigen::VectorXf& fullPose, const Eigen::Matrix3Xf& jBlend) const;
	Eigen::Matrix3Xf CalcJBlend(const SmplParam& param) const;
	Eigen::Matrix4Xf CalcChainWarps(const Eigen::Matrix4Xf& nodeWarps) const;
	Eigen::Matrix3Xf CalcJFinal(const Eigen::Matrix4Xf& chainWarps) const;
	Eigen::Matrix3Xf CalcJFinal(const SmplParam& param, const int& _jCut = -1) const;
	const Eigen::Matrix3Xi& GetFaces()const { return m_faces; }
	const Eigen::Matrix3Xf& GetVertices() const { return m_vertices; }
	Eigen::Matrix3Xf CalcVBlend(const SmplParam& param) const;
	Eigen::Matrix3Xf CalcVFinal(const SmplParam& param) const;
	Eigen::Matrix3Xf CalcVFinal(const Eigen::Matrix3Xf& jBlend, const Eigen::Matrix3Xf& vBlend, const Eigen::Matrix4Xf& chainWarps) const;
	void SaveObj(const SmplParam& param, const std::string& filename) const;
	
protected:
	SmplType m_type;
	int m_vertexSize;
	int m_faceSize;
	Eigen::Matrix3Xf m_joints;
	Eigen::MatrixXf m_jShapeBlend;
	Eigen::MatrixXf m_boneShapeBlend;
	Eigen::Matrix3Xf m_vertices;
	Eigen::Matrix3Xi m_faces;
	Eigen::MatrixXf m_jRegressor;
	Eigen::MatrixXf m_lbsWeights;
	Eigen::MatrixXf m_vShapeBlend;
	Eigen::MatrixXf m_vPoseBlend;
	Eigen::MatrixXf m_lhandComponents, m_rhandComponents;
	Eigen::VectorXf m_lhandMean, m_rhandMean;
};

