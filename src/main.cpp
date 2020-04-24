#include "smpl.h"
#include <iostream>

int main()
{
	Smpl smpl(SMPLX, "../data/smplx_male");
	SmplParam param(SMPLX);

	param.GetSomatotype().setRandom();
	param.GetSomatotype() *= 0.1f;

	param.GetExpression().setRandom();
	param.GetExpression() *= 0.1f;

	param.GetPose().setRandom();
	param.GetPose() *= 0.1f;
	param.GetLhandPCA().setRandom();
	param.GetRhandPCA().setRandom();

	smpl.SaveObj(param, "../output/test.obj");
	return 0;
}