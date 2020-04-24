#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include <filesystem>
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
	const int size = int(param.size());
	const Eigen::VectorXf tmp = param - mean.head(size);
	return ((covInv.topLeftCorner(size, size) * tmp)).dot(tmp);
}


void GaussianModel::CalcTerm(const Eigen::VectorXf& param, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb) const
{
	const int size = int(param.size());
	ATA = covInv.topLeftCorner(size, size);
	ATb = ATA * (mean.head(size) - param);
}


GaussianMixtureModel::GaussianMixtureModel(const std::string& folder)
{
	Eigen::VectorXf weight = MathUtil::LoadMat<float>((std::filesystem::path(folder) / std::filesystem::path("weight.txt")).string());
	for (int i = 0; i < weight.size(); i++)
		gaussianModels.emplace_back((std::filesystem::path(folder) / std::filesystem::path("mean_" + std::to_string(i) + ".txt")).string(),
			(std::filesystem::path(folder) / std::filesystem::path("cov_" + std::to_string(i) + ".txt")).string(), weight[i]);
}


void GaussianMixtureModel::CalcTerm(const Eigen::VectorXf& param, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb) const
{
	std::vector<float> loss;
	for (const auto& model : gaussianModels)
		loss.emplace_back(model.CalcLoss(param));

	const int idx = int(std::min_element(loss.begin(), loss.end()) - loss.begin());
	gaussianModels[idx].CalcTerm(param, ATA, ATb);
}
