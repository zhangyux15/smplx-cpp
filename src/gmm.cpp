#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include "math_util.h"
#include "gmm.h"


GaussianModel::GaussianModel(const std::string& meanFile, const std::string& covFile, const float& weight)
{
	mean = MathUtil::LoadMat<float>(meanFile);
	const Eigen::MatrixXd cov = MathUtil::LoadMat<double>(covFile);
	covInv = (0.5*log(double(weight)* 1. / (pow(2. * EIGEN_PI, 0.5*double(mean.size()))*sqrt(cov.determinant())))
		*cov.reverse()).cast<float>();
	U = covInv.llt().matrixU();
}


float GaussianModel::CalcLoss(const Eigen::VectorXf& param) const
{
	return ((covInv * (param - mean)).dot(param - mean));
}


void GaussianModel::CalcTerm(const Eigen::VectorXf& param, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb) const
{
	ATA = covInv;
	ATb = ATA * (mean - param);
}


GaussianMixtureModel::GaussianMixtureModel(const std::string& folder)
{
	Eigen::VectorXf weight = MathUtil::LoadMat<float>(folder + "/weight.txt");
	for (int i = 0; i < weight.size(); i++)
		gaussianModels.emplace_back(folder + "/mean_" + std::to_string(i) + ".txt", 
			folder + "/cov_" + std::to_string(i) + ".txt", weight[i]);
}


void GaussianMixtureModel::CalcTerm(const Eigen::VectorXf& param, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb) const
{
	std::vector<float> loss;
	for (const auto& model : gaussianModels)
		loss.emplace_back(model.CalcLoss(param));

	const int idx = int(std::min_element(loss.begin(), loss.end()) - loss.begin());
	gaussianModels[idx].CalcTerm(param, ATA, ATb);
}
