#ifndef PLAYER_H
#define PLAYER_H

//Debug Include
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\opencv.hpp"
#include "opencv2\imgproc\imgproc.hpp"

//Release Include (belum bisa)
//#include "Release\opencv.hpp"
//#include "Release\imgproc.hpp"
//#include "Release\ml.hpp"
//#include "Release\highgui.hpp"

#include "cvToBitmap.h"
#include <math.h>
#include <sstream>
using namespace cv;
using namespace std;
using namespace ml;

// accuracy
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
	assert(predicted.rows == actual.rows);
	int t = 0;
	int f = 0;
	for(int i = 0; i < actual.rows; i++) {
		float p = predicted.at<float>(i,0);
		float a = actual.at<float>(i,0);
		if((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
			t++;
		} else {
			f++;
		}
	}
	return (t * 1.0) / (t + f);
}

// plot data and class
void plot_binary(cv::Mat& data, cv::Mat& classes, string name) {
	int size = 200;
	cv::Mat plot(size, size, CV_8UC3);
	plot.setTo(cv::Scalar(255.0,255.0,255.0));
	for(int i = 0; i < data.rows; i++) {

		float x = data.at<float>(i,0) * size;
		float y = data.at<float>(i,1) * size;

		if(classes.at<float>(i, 0) > 0) {
			cv::circle(plot, cv::Point(x,y), 2, CV_RGB(255,0,0),1);
		} else {
			cv::circle(plot, cv::Point(x,y), 2, CV_RGB(0,255,0),1);
		}
	}
	cv::imshow(name, plot);
}

// function to learn
int f(float x, float y, int equation) {
	switch(equation) {
	case 0:
		return y > sin(x*10) ? -1 : 1;
		break;
	case 1:
		return y > cos(x * 10) ? -1 : 1;
		break;
	case 2:
		return y > 2*x ? -1 : 1;
		break;
	case 3:
		return y > tan(x*10) ? -1 : 1;
		break;
	default:
		return y > cos(x*10) ? -1 : 1;
	}
}

// label data with equation
cv::Mat labelData(cv::Mat points, int equation) {
	cv::Mat labels(points.rows, 1, CV_32FC1);
	for(int i = 0; i < points.rows; i++) {
		float x = points.at<float>(i,0);
		float y = points.at<float>(i,1);
		labels.at<float>(i, 0) = f(x, y, equation);
	}
	return labels;
}

// Baca data dari teks
Ptr<TrainData> BacaDataTeks(cv::String namaBerkas, int lokasiKolomOutput, int panjangKolomOutput) {
	//Untuk masukan data ini disarankan data sudah diatur dulu, sehingga tidak ada data input yang di skip
	Ptr<TrainData> data;
	data = TrainData::loadFromCSV(namaBerkas,0,lokasiKolomOutput, panjangKolomOutput, cv::String(), ';', '?');
	return data;
}

//Fungsi untuk melatih data menggunakan Support Vector Machine (belum diuji)
Mat svm(bool tipeProses, Ptr<TrainData> data, Ptr<TrainData> dataUji, Mat konfigurasi, cv::String model) {
	int size = 200;
	float errorDataTes, errorDataUji;
	if (!tipeProses) {
		int svmTipe = konfigurasi.at<int>(0,0);
		int svmKernel = konfigurasi.at<int>(0,1);		
		double svmDegree = konfigurasi.at<double>(0,2);
		double svmGamma = konfigurasi.at<double>(0,3);
		double svmCoef0 = konfigurasi.at<double>(0,4);
		double svmC = konfigurasi.at<double>(0,5);
		double svmNu = konfigurasi.at<double>(0,6);
		double svmP = konfigurasi.at<double>(0,7);

		Ptr<SVM> svmfunc = SVM::create();

		svmfunc->setType(svmTipe); 	
		svmfunc->setKernel(svmKernel); //CvSVM::RBF, CvSVM::LINEAR ...
		svmfunc->setDegree(svmDegree); // for poly
		svmfunc->setGamma(svmGamma); // for poly/rbf/sigmoid
		svmfunc->setCoef0(svmCoef0); // for poly/sigmoid

		svmfunc->setC(svmC); // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
		svmfunc->setNu(svmNu); // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
		svmfunc->setP(svmP); // for CV_SVM_EPS_SVR	

		//svmfunc->setClassWeights(); // for CV_SVM_C_SVC
		//svmfunc->setTermCriteria = CV_TERMCRIT_ITER +CV_TERMCRIT_EPS;
		svmfunc->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));

		// SVM training (use train auto for OpenCV>=2.0)
		//Ptr<TrainData> td = TrainData::create(trainingData, ROW_SAMPLE, trainingClasses);
		svmfunc->train(data);
		Mat respon;
		errorDataTes = svmfunc->calcError(data, false, respon);
		errorDataUji = svmfunc->calcError(dataUji, false, respon);

		//Penyimpanan algoritma
		stringstream teks;
		cv::String nama = "SVM Error_";
		cv::String tipe = ".csv";
		cv::String namaSimpan;
		teks<<nama<<errorDataTes<<tipe;
		namaSimpan = teks.str();
		svmfunc->save(namaSimpan);

		// plot support vectors
		cv::Mat plot_sv(size, size, CV_8UC3);
		plot_sv.setTo(cv::Scalar(255.0,255.0,255.0));
		Mat sv = svmfunc->getSupportVectors();
		for (int i = 0; i < sv.rows; ++i)
		{
			const float* v = sv.ptr<float>(i);
			circle(plot_sv, cv::Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128));
		}
		cv::imshow("Support Vectors", plot_sv);
	}
	else if (tipeProses) {
		Mat respon;
		Ptr<SVM> svmfunc = Algorithm::load<SVM>(model);
		errorDataTes = svmfunc->calcError(data, false, respon);
		errorDataUji = svmfunc->calcError(dataUji, false, respon);
	}
	Mat dataError(2,1,CV_32F);	
	dataError.at<float>(0,0) = errorDataTes;
	dataError.at<float>(1,0) = errorDataUji;
	return dataError;
}

//Fungsi untuk melatih data menggunakan Support Vector Machine (sudah berfungsi)
Mat mlp(bool tipeProses, Ptr<TrainData> data, Ptr<TrainData> dataUji, Mat layersKonfig, Mat konfigurasi, cv::String model) {
	//Ekstraksi konfigurasi layer dari matriks layersKonfig yang sudah ditentukan pengguna
	int lyrDepan = (int) layersKonfig.at<float>(0,0);
	int lyrOut = (int) layersKonfig.at<float>(0,1);
	int lyrHid1 = (int) layersKonfig.at<float>(0,2);
	int lyrHid2 = (int) layersKonfig.at<float>(0,3);
	int lyrHid3 = (int) layersKonfig.at<float>(0,4);
	int lyrHid4 = (int) layersKonfig.at<float>(0,5);
	int lyrHid5 = (int) layersKonfig.at<float>(0,6);
	int banyakHdnLayer = (int) layersKonfig.at<float>(0,7);

	//Ekstraksi konfigurasi mlp dari matriks konfigurasi yang sudah ditentukan pengguna
	int maxIter = (int) konfigurasi.at<float>(0,0);
	double epsilon = (double) konfigurasi.at<float>(0,1);
	int trainMethod = (int) konfigurasi.at<float>(0,2);
	double bpWeightScale = (double) konfigurasi.at<float>(0,3);
	double bpMomentumScale = (double) konfigurasi.at<float>(0,4);
	double rpDW0 = (double) konfigurasi.at<float>(0,5);
	double rpDWMin = (double) konfigurasi.at<float>(0,6);
	int activateFunc = (int) konfigurasi.at<float>(0,7);
	double alpha = (double) konfigurasi.at<float>(0,8);
	double beta = (double) konfigurasi.at<float>(0,9);

	float errorDataTes, errorDataUji;

	//Pembuatan skema layer (input, hidden, dan output)
	cv::Mat layers = Mat(banyakHdnLayer+2, 1, CV_32SC1);
	layers.row(0) = cv::Scalar(lyrDepan);
	layers.row(banyakHdnLayer+1) = cv::Scalar(lyrOut);
	switch (banyakHdnLayer) {
	case 1:					
		layers.row(1) = cv::Scalar(lyrHid1);		
		break;
	case 2:		
		layers.row(1) = cv::Scalar(lyrHid1);
		layers.row(2) = cv::Scalar(lyrHid2);		
		break;
	case 3:		
		layers.row(1) = cv::Scalar(lyrHid1);
		layers.row(2) = cv::Scalar(lyrHid2);
		layers.row(3) = cv::Scalar(lyrHid3);		
		break;
	case 4:		
		layers.row(1) = cv::Scalar(lyrHid1);
		layers.row(2) = cv::Scalar(lyrHid2);
		layers.row(3) = cv::Scalar(lyrHid3);
		layers.row(4) = cv::Scalar(lyrHid4);		
		break;
	case 5:		
		layers.row(1) = cv::Scalar(lyrHid1);
		layers.row(2) = cv::Scalar(lyrHid2);
		layers.row(3) = cv::Scalar(lyrHid3);		
		layers.row(4) = cv::Scalar(lyrHid4);
		layers.row(5) = cv::Scalar(lyrHid5);
		break;
	}

	//@tipeProses adalah variabel untuk menentukan jenis operasi yang dilakukan fungsi mlp()
	//Jika false, maka mlp() akan melakukan pelatihan dengan data latih dari awal
	//Jika true, maka mlp() akan melakukan pengujian dengan algoritma yang sudah dilatih sebelumnya
	if (!tipeProses) {
		Ptr<ANN_MLP> mlpfunc = ANN_MLP::create();	
		CvTermCriteria criteria;
		criteria.max_iter = maxIter; //biasanya 100
		criteria.epsilon = epsilon; //biasanya 0.0001f
		criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
		mlpfunc->setLayerSizes(layers);
		mlpfunc->setTrainMethod(trainMethod); //BACKPROP = 0, RPROP = 1
		mlpfunc->setBackpropWeightScale(bpWeightScale);
		mlpfunc->setBackpropMomentumScale(bpMomentumScale);
		mlpfunc->setRpropDWMax(50);
		mlpfunc->setRpropDWMinus(0.5);
		mlpfunc->setRpropDWPlus(1.2);
		mlpfunc->setRpropDW0(rpDW0);
		mlpfunc->setRpropDWMin(rpDWMin);		
		mlpfunc->setTermCriteria(criteria);		
		mlpfunc->setActivationFunction(activateFunc, alpha, beta); //Identity = 0, Sigmoid = 1, Gaussian = 2		

		// Mulai latih data
		Mat respon; //Variabel mainan, tidak digunakan
		mlpfunc->train(data);	

		errorDataTes = mlpfunc->calcError(data, false, respon);
		errorDataUji = mlpfunc->calcError(dataUji, false, respon);

		//Penyimpanan algoritma hasil pelatihan
		stringstream teks;
		cv::String nama = "MLP konf_";
		cv::String pisah = "_";
		cv::String tipe = ".csv";
		cv::String namaSimpan;
		cv::String activ;
		if (activateFunc == 0)
			activ = "Identity";
		else if (activateFunc == 1)
			activ = "Sigmoid";
		else
			activ = "Gaussian";

		if (trainMethod == 0) {			
			teks<<nama<<"BackProp_hdnLayer_"<<banyakHdnLayer<<"_Activ "<<activ<<" "<<alpha<<" "<<beta<<"_Weight_"<<bpWeightScale<<"_Momentum_"<<bpMomentumScale<<"_Iter_"<<maxIter<<" error "<<errorDataTes<<tipe;
		}
		else if (trainMethod == 1) {
			teks<<nama<<"Prop_hdnLayer_"<<banyakHdnLayer<<"_Activ "<<activ<<" "<<alpha<<" "<<beta<<"_DW0_"<<rpDW0<<"_DWMin_"<<rpDWMin<<"_Iter_"<<maxIter<<" error "<<errorDataTes<<tipe;
		}		
		namaSimpan = teks.str();
		mlpfunc->save(namaSimpan);
	}

	else if (tipeProses) {
		Mat respon;
		Ptr<ANN_MLP> mlpfunc = Algorithm::load<ANN_MLP>(model);	
		errorDataTes = 0.0;
		errorDataUji = mlpfunc->calcError(dataUji, false, respon);
	}
	Mat dataError(2,1,CV_32F);	
	dataError.at<float>(0,0) = errorDataTes;
	dataError.at<float>(1,0) = errorDataUji;
	return dataError;
}

//Fungsi untuk melatih data menggunakan K-Nearest Neighbour (belum berfungsi)
void knn(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses, int K) {
	Ptr<KNearest> knnfunc = KNearest::create();	
	Ptr<TrainData> td = TrainData::create(trainingData, ROW_SAMPLE, trainingClasses);
	knnfunc->train(td);
	cv::Mat predicted(testClasses.rows, 1, CV_32F);
	for(int i = 0; i < testData.rows; i++) {
		const cv::Mat sample = testData.row(i);		
		knnfunc->findNearest(sample, K, predicted);
	}
	plot_binary(testData, predicted, "Predictions KNN");
}

//Fungsi untuk melatih data menggunakan Bayes (belum berfungsi)
void bayes(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {
	Ptr<NormalBayesClassifier> bayesfunc = NormalBayesClassifier::create();	
	Ptr<TrainData> td = TrainData::create(trainingData, ROW_SAMPLE, trainingClasses);
	bayesfunc->train(td);	
	cv::Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		const cv::Mat sample = testData.row(i);
		bayesfunc->predict(sample, predicted);		
	}
	plot_binary(testData, predicted, "Predictions Bayes");
}

//Fungsi untuk melatih data menggunakan Decision Tree (belum berfungsi)
void decisiontree(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {
	Ptr<DTrees> dtreefunc = DTrees::create();	
	Ptr<TrainData> td = TrainData::create(trainingData, ROW_SAMPLE, trainingClasses);	
	cv::Mat var_type(3, 1, CV_8U);

	// define attributes as numerical
	var_type.at<unsigned int>(0,0) = VAR_NUMERICAL;
	var_type.at<unsigned int>(0,1) = VAR_NUMERICAL;
	// define output node as numerical
	var_type.at<unsigned int>(0,2) = VAR_NUMERICAL;

	dtreefunc->train(td);	
	cv::Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		const cv::Mat sample = testData.row(i);
		dtreefunc->predict(sample,predicted);		
	}
	plot_binary(testData, predicted, "Predictions tree");
}

#endif