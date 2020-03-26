#pragma once
#include <Eigen/Core>
#include <string>
#include <vector>


struct GaussianModel
{
	Eigen::VectorXf mean;
	Eigen::MatrixXf covInv;
	Eigen::MatrixXf U;

	GaussianModel() {}
	GaussianModel(const std::string& meanFile, const std::string& covFile, const float& weight = 1.f);
	float CalcLoss(const Eigen::VectorXf& param) const;
	void CalcTerm(const Eigen::VectorXf& param, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb) const;
};


struct GaussianMixtureModel
{
	std::vector<GaussianModel> gaussianModels;

	GaussianMixtureModel(const std::string& folder);
	void CalcTerm(const Eigen::VectorXf& param, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb) const;
};

