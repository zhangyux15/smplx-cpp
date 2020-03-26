#pragma once
#include <Eigen/Core>
#include <string>
#include <memory>
#include "smpl.h"
#include "gmm.h"
#include "smpl.h"
#include "math_util.h"


class SmplSolver : public Smpl
{
public:
	struct Status
	{
		SmplParam param;
		Eigen::VectorXf fullPose;
		Eigen::Matrix3Xf jBlend, jFinal,vBlend, vFinal;
		Eigen::Matrix4Xf nodeWarps, chainWarps;
		Eigen::MatrixXf nodeWarpsJacobi;
		Eigen::MatrixXf jShapeJacobi, jPoseJacobi, vShapeJacobi, vPoseJacobi, jFullPoseJacobi, vFullPoseJacobi;
		Eigen::MatrixXf shapeATA, poseATA;
		Eigen::VectorXf shapeATb, poseATb;
		Eigen::VectorXf validVMask;
		int jCut;
		bool validVertex;
	};

	struct BasicTerm
	{
		BasicTerm(const float& _w) :w(_w) {}
		float w;
		virtual void ConfigMask(Eigen::VectorXf& validVMask) const {}
		virtual void CalcTerm(Status& status) const = 0;
	};


	struct J3dTerm : public BasicTerm
	{
		Eigen::Matrix4Xf jTarget;				// last row is weight
		J3dTerm(const Eigen::Matrix4Xf& _jTarget, const float& _w = 1.f) : BasicTerm(_w), jTarget(_jTarget) {}
		void CalcTerm(Status& status) const override;
	};


	struct J2dTerm : public BasicTerm
	{
		std::vector<Eigen::Matrix3Xf> jTarget2d;
		std::vector<Eigen::Matrix34f> projs;
		J2dTerm(const std::vector<Eigen::Matrix3Xf>& _jTarget2d, const std::vector<Eigen::Matrix34f>& _projs, const float& _w = 1.f)
			:BasicTerm(_w), jTarget2d(_jTarget2d), projs(_projs) {}
		void CalcTerm(Status& status) const override;
	};


	struct TemporalTerm : public BasicTerm
	{
		SmplParam paramPrev;
		float wTrans, wPoseBody, wPoseHand, wSomatotype, wExpression;

		TemporalTerm(const SmplParam& _paramPrev, const float& _wTrans, const float& _wPoseBody, const float& _wPoseHand, 
			const float& _wSomatotype, const float& _wExpression, const float& _w = 1.f)
			: BasicTerm(_w), paramPrev(_paramPrev), wTrans(_wTrans), wPoseBody(_wPoseBody), wPoseHand(_wPoseHand),
			wSomatotype(_wSomatotype), wExpression(_wExpression) {}
		void CalcTerm(Status& status) const override;
	};


	struct V3dTerm : public BasicTerm
	{
		Eigen::Matrix4Xf vTarget;
		V3dTerm(const Eigen::Matrix4Xf& _vTarget, const float& _w = 1.f) : BasicTerm(_w), vTarget(_vTarget) {}
		void ConfigMask(Eigen::VectorXf& validVMask) const override;
		void CalcTerm(Status& status) const override;
	};


	struct V2dTerm : public BasicTerm
	{
		std::vector<Eigen::Matrix3Xf> vTarget2d;
		std::vector<Eigen::Matrix34f> projs;
		V2dTerm(const std::vector<Eigen::Matrix3Xf>& _vTarget2d, const std::vector<Eigen::Matrix34f>& _projs, const float& _w = 1.f)
			:BasicTerm(_w), vTarget2d(_vTarget2d), projs(_projs) {}
		void ConfigMask(Eigen::VectorXf& validVMask) const override;
		void CalcTerm(Status& status) const override;
	};


	struct PriorTerm : public BasicTerm
	{
		float wPoseBody, wPoseHand, wSomatotype, wExpression;
		GaussianMixtureModel bodyGmm;
		PriorTerm(const std::string& gmmFolder, const float& _wPoseBody, const float& _wPoseHand,
			const float& _wSomatotype, const float& _wExpression, const float& _w = 1.f)
			:BasicTerm(_w), wPoseBody(_wPoseBody), wPoseHand(_wPoseHand), wSomatotype(_wSomatotype), wExpression(_wExpression), bodyGmm(gmmFolder) {}
		void CalcTerm(Status& status) const override;
	};

	struct RegularTerm : public BasicTerm
	{
		float wPoseBody, wPoseHand, wSomatotype, wExpression;
		RegularTerm(const float& _wPoseBody, const float& _wPoseHand, const float& _wSomatotype, const float& _wExpression, const float& _w = 1.f)
			:BasicTerm(_w), wPoseBody(_wPoseBody), wPoseHand(_wPoseHand), wSomatotype(_wSomatotype), wExpression(_wExpression) {}
		void CalcTerm(Status& status) const override;
	};

	SmplSolver(const SmplType& _type, const std::string &modelPath) :Smpl(_type, modelPath){}
	~SmplSolver() = default;

	void SolveShape(const std::vector<std::shared_ptr<BasicTerm>>& terms, SmplParam& param, const int& maxIterTime, const float& updateThresh = 1e-5f) const;
	void SolvePose(const std::vector<std::shared_ptr<BasicTerm>>& terms, SmplParam& param, const int& maxIterTime, const float& updateThresh = 1e-5f) const;
	void SolvePoseHier(const std::vector<std::shared_ptr<BasicTerm>>& terms, SmplParam& param, const int& maxIterTime, const float& updateThresh = 1e-5f) const;

	void AlignShape(const Eigen::Matrix3Xf& jTarget, SmplParam& param, std::shared_ptr<SmplSolver::PriorTerm> priorTerm, std::shared_ptr<SmplSolver::RegularTerm> regularTerm, const int& iterTime, const float& updateThresh = 1e-5f) const;
	void AlignPose(const Eigen::Matrix3Xf& jTarget, SmplParam& param, std::shared_ptr<SmplSolver::PriorTerm> priorTerm, std::shared_ptr<SmplSolver::RegularTerm> regularTerm, const int& iterTime, const float& updateThresh = 1e-5f) const;
	void AlignRT(const Eigen::Matrix3Xf& jTarget, SmplParam& param) const;

protected:
	Status InitStatus(const std::vector<std::shared_ptr<BasicTerm>>& terms, const bool& shapeValid, const bool& poseValid) const;
	void CalcStatus(const SmplParam& param, Status& status, const int& _jCut) const;
};