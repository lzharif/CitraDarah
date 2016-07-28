#pragma once

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
#include "HeaderProgramLatih.h"

using namespace cv;
using namespace std;
using namespace ml;

//Variabel Global untuk Tahap Pelatihan
int varInput, varOutput;
bool statusUji = false, statusInputLatih = false, statusInputUji = false;
//Pelatihan menggunakan MLP
int inputLayer, outputLayer, hdnLayer1, hdnLayer2, hdnLayer3, hdnLayer4, hdnLayer5, bykHdnLayer, indexTrain, indexFAktif, iterasiMLP;
double epsilon, dw0, dwMin, alfa, beta;
cv::String modelUji;

Ptr<TrainData> dataLatih, dataUji; //Data latih sudah termasuk data luaran

//Variabel Global untuk Tahap Ekstraksi
#define PI 3.14159265358979323846

Mat citraAwal;
int h = 1;
int luasPlasma = 0, luasInti = 0, kelilingPlasma = 0, kelilingInti = 0;
float solidityPlasma = 0.0, solidityInti = 0.0, circularityPlasma = 0.0, circularityInti = 0.0, nilaiStddevPlasma = 0.0, nilaiStddevInti = 0.0, liLP = 0.0, kiKP = 0.0, luasNormalInti = 0.0, kelilingNormalInti = 0.0, eccentricity = 0.0, entropi = 0.0, energi = 0.0, kontras = 0.0, homogenitas = 0.0;
int simpanLuasPlasma[108], simpanLuasInti[108], simpanKelilingPlasma[108], simpanKelilingInti[108];
float simpanSolidityPlasma[108], simpanSolidityInti[108], simpanCircularityPlasma[108], simpanCircularityInti[108], simpanStddevPlasma[108], simpanStddevInti[108], simpanLILP[108], simpanKIKP[108], simpanLuasNormalInti[108], simpanKelilingNormalInti[108], simpanEccentricity[108];
//int tPlasmaH = 81, tPlasmaS = 28, tPlasmaV = 74, tIntiS = 81;
int tPlasmaH = 130, tPlasmaS = 27, tPlasmaV = 62, tIntiS = 149;
int temp_tPlasmaH = tPlasmaH, temp_tPlasmaS = tPlasmaS, temp_tPlasmaV = tPlasmaV, temp_tIntiS = tIntiS;
char* namaBerkasCitra;
char namaBerkasCitraFix[60];
FILE *outfile;


namespace CitraDarah {

	using namespace System;
	using namespace System::IO;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::StatusStrip^  statusStrip1;
	protected:
	private: System::Windows::Forms::ToolStripStatusLabel^  toolStripStatusLabel1;
	private: System::Windows::Forms::ToolStripStatusLabel^  toolStripStatusProgram;
	private: System::Windows::Forms::MenuStrip^  menuStrip1;
	private: System::Windows::Forms::ToolStripMenuItem^  berkasToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  muatCitraEkstraksiToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  olahBanyakEkstraksiToolStripMenuItem;


	private: System::Windows::Forms::ToolStripMenuItem^  simpanToolStripMenuItem;
	private: System::Windows::Forms::OpenFileDialog^  openFileDialogTeks;
	private: System::Windows::Forms::TabControl^  tabControl1;
	private: System::Windows::Forms::TabPage^  tabPageEkstraksiFitur;
	private: System::Windows::Forms::TabPage^  tabPageLatih;


	private: System::Windows::Forms::Label^  labelErrorDataUji;
	private: System::Windows::Forms::Label^  labelErrorDataTes;
	private: System::Windows::Forms::Label^  label40;
	private: System::Windows::Forms::Label^  label39;
	private: System::Windows::Forms::Label^  label36;
	private: System::Windows::Forms::CheckBox^  checkBoxUjiPelatihan;
	private: System::Windows::Forms::GroupBox^  groupBoxMLP;
	private: System::Windows::Forms::Label^  labelDWMin;
	private: System::Windows::Forms::TextBox^  textBoxDWMin;
	private: System::Windows::Forms::Label^  labelDW0;
	private: System::Windows::Forms::TextBox^  textBoxDW0;
	private: System::Windows::Forms::Label^  labelMomentum;
	private: System::Windows::Forms::TextBox^  textBoxMomentum;
	private: System::Windows::Forms::Label^  labelWeight;
	private: System::Windows::Forms::TextBox^  textBoxWeight;
	private: System::Windows::Forms::Label^  label44;
	private: System::Windows::Forms::TextBox^  textBoxEpsilon;
	private: System::Windows::Forms::Label^  label45;
	private: System::Windows::Forms::TextBox^  textBoxIterasiMLP;
	private: System::Windows::Forms::ComboBox^  comboBoxActivationFunc;
	private: System::Windows::Forms::ComboBox^  comboBoxTrainMethod;
	private: System::Windows::Forms::Label^  label43;
	private: System::Windows::Forms::Label^  label42;
	private: System::Windows::Forms::NumericUpDown^  numericUpDownBykHdn;
	private: System::Windows::Forms::Label^  labelHdn5;
	private: System::Windows::Forms::NumericUpDown^  numericUpDownHdn5;
	private: System::Windows::Forms::Label^  labelHdn4;
	private: System::Windows::Forms::NumericUpDown^  numericUpDownHdn4;
	private: System::Windows::Forms::Label^  labelHdn3;
	private: System::Windows::Forms::NumericUpDown^  numericUpDownHdn3;
	private: System::Windows::Forms::Label^  labelHdn2;
	private: System::Windows::Forms::NumericUpDown^  numericUpDownHdn2;
	private: System::Windows::Forms::Label^  label22;
	private: System::Windows::Forms::NumericUpDown^  numericUpDownOutput;
	private: System::Windows::Forms::Label^  label19;
	private: System::Windows::Forms::NumericUpDown^  numericUpDownHdn1;
	private: System::Windows::Forms::Label^  label18;
	private: System::Windows::Forms::Label^  label11;
	private: System::Windows::Forms::NumericUpDown^  numericUpDownInputLayer;
	private: System::Windows::Forms::Label^  label16;
	private: System::Windows::Forms::TextBox^  textBoxBeta;
	private: System::Windows::Forms::Label^  label17;
	private: System::Windows::Forms::TextBox^  textBoxAlpha;
	private: System::Windows::Forms::GroupBox^  groupBoxSVM;
	private: System::Windows::Forms::Label^  label14;
	private: System::Windows::Forms::TextBox^  textBox3;
	private: System::Windows::Forms::Label^  label15;
	private: System::Windows::Forms::TextBox^  textBox4;
	private: System::Windows::Forms::GroupBox^  groupBoxSupervisedData;
	private: System::Windows::Forms::Label^  label13;
	private: System::Windows::Forms::TextBox^  textBoxBanyakOutput;
	private: System::Windows::Forms::Label^  label12;
	private: System::Windows::Forms::TextBox^  textBoxBanyakInput;
	private: System::Windows::Forms::Button^  buttonMulaiPelatihan;
	private: System::Windows::Forms::Label^  labelLangkah3;
	private: System::Windows::Forms::Label^  label10;
	private: System::Windows::Forms::RadioButton^  radioButtonDTree;
	private: System::Windows::Forms::RadioButton^  radioButtonBayes;
	private: System::Windows::Forms::RadioButton^  radioButtonKNN;
	private: System::Windows::Forms::RadioButton^  radioButtonMLP;
	private: System::Windows::Forms::RadioButton^  radioButtonLatihSVM;
	private: System::Windows::Forms::Label^  label7;
	private: System::Windows::Forms::Label^  label8;
	private: System::Windows::Forms::Label^  labelDataUji;
	private: System::Windows::Forms::Label^  labelDataLatih;
	private: System::Windows::Forms::Label^  label6;
	private: System::Windows::Forms::Button^  buttonBukaDataUji;
	private: System::Windows::Forms::Button^  buttonBukaDataLatih;
	private: System::Windows::Forms::Label^  label2;
	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::GroupBox^  groupBox3;
	private: System::Windows::Forms::PictureBox^  pictureBoxRGB;
	private: System::Windows::Forms::PictureBox^  pictureBoxSitoplasma;
	private: System::Windows::Forms::Label^  label48;
	private: System::Windows::Forms::PictureBox^  pictureBoxInti;
	private: System::Windows::Forms::Label^  label47;
	private: System::Windows::Forms::Label^  label46;
	private: System::Windows::Forms::GroupBox^  groupBox2;
	private: System::Windows::Forms::Label^  label5;
	private: System::Windows::Forms::TextBox^  textBoxIntiS;
	private: System::Windows::Forms::HScrollBar^  hScrollBarIntiS;
	private: System::Windows::Forms::Label^  label9;
	private: System::Windows::Forms::Label^  label21;
	private: System::Windows::Forms::TextBox^  textBoxPlasmaV;
	private: System::Windows::Forms::HScrollBar^  hScrollBarPlasmaV;
	private: System::Windows::Forms::Label^  label23;
	private: System::Windows::Forms::TextBox^  textBoxPlasmaS;
	private: System::Windows::Forms::HScrollBar^  hScrollBarPlasmaS;
	private: System::Windows::Forms::Label^  label24;
	private: System::Windows::Forms::TextBox^  textBoxPlasmaH;
	private: System::Windows::Forms::HScrollBar^  hScrollBarPlasmaH;
	private: System::Windows::Forms::Label^  label25;
	private: System::Windows::Forms::GroupBox^  groupBox1;
	private: System::Windows::Forms::RadioButton^  radioButtonConvexArea;
	private: System::Windows::Forms::RadioButton^  radioButtonConvexHull;
	private: System::Windows::Forms::RadioButton^  radioButtonKontur;
	private: System::Windows::Forms::RadioButton^  radioButtonBiner;












	private: System::Windows::Forms::GroupBox^  groupBox4;
	private: System::Windows::Forms::TextBox^  textBoxLuasNormalisasiInti;
	private: System::Windows::Forms::Label^  label38;
	private: System::Windows::Forms::TextBox^  textBoxEccentricity;
	private: System::Windows::Forms::TextBox^  textBoxKelilingNormalisasiInti;
	private: System::Windows::Forms::Label^  label35;
	private: System::Windows::Forms::Label^  label37;
	private: System::Windows::Forms::TextBox^  textBoxKelilingIntiPlasma;
	private: System::Windows::Forms::TextBox^  textBoxLuasIntiPlasma;
	private: System::Windows::Forms::Label^  label4;
	private: System::Windows::Forms::Label^  label26;

	private: System::Windows::Forms::TextBox^  textBoxHomogenityPlasma;
	private: System::Windows::Forms::Label^  label51;
	private: System::Windows::Forms::TextBox^  textBoxContrastPlasma;
	private: System::Windows::Forms::Label^  label52;
	private: System::Windows::Forms::Label^  label20;
	private: System::Windows::Forms::TextBox^  textBoxEnergyPlasma;
	private: System::Windows::Forms::TextBox^  textBoxEntropyPlasma;
	private: System::Windows::Forms::TextBox^  textBoxBPlasma;
	private: System::Windows::Forms::TextBox^  textBoxGPlasma;
	private: System::Windows::Forms::TextBox^  textBoxRPlasma;
	private: System::Windows::Forms::Label^  label27;
	private: System::Windows::Forms::Label^  label28;
	private: System::Windows::Forms::Label^  label29;
	private: System::Windows::Forms::Label^  label30;
	private: System::Windows::Forms::Label^  label31;
	private: System::Windows::Forms::TextBox^  textBoxCircularityPlasma;
	private: System::Windows::Forms::TextBox^  textBoxGranularityPlasma;
	private: System::Windows::Forms::TextBox^  textBoxSolidityPlasma;
	private: System::Windows::Forms::TextBox^  textBoxKelilingPlasma;
	private: System::Windows::Forms::TextBox^  textBoxLuasPlasma;
	private: System::Windows::Forms::Label^  label32;
	private: System::Windows::Forms::Label^  label33;
	private: System::Windows::Forms::Label^  label34;
	private: System::Windows::Forms::Label^  label41;
	private: System::Windows::Forms::Label^  label49;
	private: System::Windows::Forms::TextBox^  textBoxHomogenityInti;
	private: System::Windows::Forms::Label^  label63;
	private: System::Windows::Forms::TextBox^  textBoxContrastInti;
	private: System::Windows::Forms::Label^  label64;
	private: System::Windows::Forms::TextBox^  textBoxEnergyInti;
	private: System::Windows::Forms::Label^  label58;
	private: System::Windows::Forms::TextBox^  textBoxEntropyInti;
	private: System::Windows::Forms::TextBox^  textBoxBInti;
	private: System::Windows::Forms::TextBox^  textBoxGInti;
	private: System::Windows::Forms::TextBox^  textBoxRInti;
	private: System::Windows::Forms::Label^  label59;
	private: System::Windows::Forms::Label^  label60;
	private: System::Windows::Forms::Label^  label61;
	private: System::Windows::Forms::Label^  label62;
	private: System::Windows::Forms::TextBox^  textBoxCircularityInti;
	private: System::Windows::Forms::Label^  label50;
	private: System::Windows::Forms::TextBox^  textBoxGranularityInti;
	private: System::Windows::Forms::TextBox^  textBoxSolidityInti;
	private: System::Windows::Forms::TextBox^  textBoxKelilingInti;
	private: System::Windows::Forms::TextBox^  textBoxLuasInti;
	private: System::Windows::Forms::Label^  label53;
	private: System::Windows::Forms::Label^  label54;
	private: System::Windows::Forms::Label^  label55;
	private: System::Windows::Forms::Label^  label56;
	private: System::Windows::Forms::Label^  label57;
	private: System::Windows::Forms::FolderBrowserDialog^  folderBrowserDialog;
	private: System::Windows::Forms::OpenFileDialog^  openFileDialogCitra;

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(MyForm::typeid));
			this->statusStrip1 = (gcnew System::Windows::Forms::StatusStrip());
			this->toolStripStatusLabel1 = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->toolStripStatusProgram = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->berkasToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->muatCitraEkstraksiToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->olahBanyakEkstraksiToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->simpanToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openFileDialogTeks = (gcnew System::Windows::Forms::OpenFileDialog());
			this->tabControl1 = (gcnew System::Windows::Forms::TabControl());
			this->tabPageLatih = (gcnew System::Windows::Forms::TabPage());
			this->labelErrorDataUji = (gcnew System::Windows::Forms::Label());
			this->labelErrorDataTes = (gcnew System::Windows::Forms::Label());
			this->label40 = (gcnew System::Windows::Forms::Label());
			this->label39 = (gcnew System::Windows::Forms::Label());
			this->label36 = (gcnew System::Windows::Forms::Label());
			this->checkBoxUjiPelatihan = (gcnew System::Windows::Forms::CheckBox());
			this->groupBoxMLP = (gcnew System::Windows::Forms::GroupBox());
			this->labelDWMin = (gcnew System::Windows::Forms::Label());
			this->textBoxDWMin = (gcnew System::Windows::Forms::TextBox());
			this->labelDW0 = (gcnew System::Windows::Forms::Label());
			this->textBoxDW0 = (gcnew System::Windows::Forms::TextBox());
			this->labelMomentum = (gcnew System::Windows::Forms::Label());
			this->textBoxMomentum = (gcnew System::Windows::Forms::TextBox());
			this->labelWeight = (gcnew System::Windows::Forms::Label());
			this->textBoxWeight = (gcnew System::Windows::Forms::TextBox());
			this->label44 = (gcnew System::Windows::Forms::Label());
			this->textBoxEpsilon = (gcnew System::Windows::Forms::TextBox());
			this->label45 = (gcnew System::Windows::Forms::Label());
			this->textBoxIterasiMLP = (gcnew System::Windows::Forms::TextBox());
			this->comboBoxActivationFunc = (gcnew System::Windows::Forms::ComboBox());
			this->comboBoxTrainMethod = (gcnew System::Windows::Forms::ComboBox());
			this->label43 = (gcnew System::Windows::Forms::Label());
			this->label42 = (gcnew System::Windows::Forms::Label());
			this->numericUpDownBykHdn = (gcnew System::Windows::Forms::NumericUpDown());
			this->labelHdn5 = (gcnew System::Windows::Forms::Label());
			this->numericUpDownHdn5 = (gcnew System::Windows::Forms::NumericUpDown());
			this->labelHdn4 = (gcnew System::Windows::Forms::Label());
			this->numericUpDownHdn4 = (gcnew System::Windows::Forms::NumericUpDown());
			this->labelHdn3 = (gcnew System::Windows::Forms::Label());
			this->numericUpDownHdn3 = (gcnew System::Windows::Forms::NumericUpDown());
			this->labelHdn2 = (gcnew System::Windows::Forms::Label());
			this->numericUpDownHdn2 = (gcnew System::Windows::Forms::NumericUpDown());
			this->label22 = (gcnew System::Windows::Forms::Label());
			this->numericUpDownOutput = (gcnew System::Windows::Forms::NumericUpDown());
			this->label19 = (gcnew System::Windows::Forms::Label());
			this->numericUpDownHdn1 = (gcnew System::Windows::Forms::NumericUpDown());
			this->label18 = (gcnew System::Windows::Forms::Label());
			this->label11 = (gcnew System::Windows::Forms::Label());
			this->numericUpDownInputLayer = (gcnew System::Windows::Forms::NumericUpDown());
			this->label16 = (gcnew System::Windows::Forms::Label());
			this->textBoxBeta = (gcnew System::Windows::Forms::TextBox());
			this->label17 = (gcnew System::Windows::Forms::Label());
			this->textBoxAlpha = (gcnew System::Windows::Forms::TextBox());
			this->groupBoxSVM = (gcnew System::Windows::Forms::GroupBox());
			this->label14 = (gcnew System::Windows::Forms::Label());
			this->textBox3 = (gcnew System::Windows::Forms::TextBox());
			this->label15 = (gcnew System::Windows::Forms::Label());
			this->textBox4 = (gcnew System::Windows::Forms::TextBox());
			this->groupBoxSupervisedData = (gcnew System::Windows::Forms::GroupBox());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->label13 = (gcnew System::Windows::Forms::Label());
			this->textBoxBanyakOutput = (gcnew System::Windows::Forms::TextBox());
			this->label12 = (gcnew System::Windows::Forms::Label());
			this->textBoxBanyakInput = (gcnew System::Windows::Forms::TextBox());
			this->buttonMulaiPelatihan = (gcnew System::Windows::Forms::Button());
			this->labelLangkah3 = (gcnew System::Windows::Forms::Label());
			this->label10 = (gcnew System::Windows::Forms::Label());
			this->radioButtonDTree = (gcnew System::Windows::Forms::RadioButton());
			this->radioButtonBayes = (gcnew System::Windows::Forms::RadioButton());
			this->radioButtonKNN = (gcnew System::Windows::Forms::RadioButton());
			this->radioButtonMLP = (gcnew System::Windows::Forms::RadioButton());
			this->radioButtonLatihSVM = (gcnew System::Windows::Forms::RadioButton());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->label8 = (gcnew System::Windows::Forms::Label());
			this->labelDataUji = (gcnew System::Windows::Forms::Label());
			this->labelDataLatih = (gcnew System::Windows::Forms::Label());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->buttonBukaDataUji = (gcnew System::Windows::Forms::Button());
			this->buttonBukaDataLatih = (gcnew System::Windows::Forms::Button());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->tabPageEkstraksiFitur = (gcnew System::Windows::Forms::TabPage());
			this->groupBox4 = (gcnew System::Windows::Forms::GroupBox());
			this->textBoxHomogenityInti = (gcnew System::Windows::Forms::TextBox());
			this->label63 = (gcnew System::Windows::Forms::Label());
			this->textBoxContrastInti = (gcnew System::Windows::Forms::TextBox());
			this->label64 = (gcnew System::Windows::Forms::Label());
			this->textBoxEnergyInti = (gcnew System::Windows::Forms::TextBox());
			this->label58 = (gcnew System::Windows::Forms::Label());
			this->textBoxEntropyInti = (gcnew System::Windows::Forms::TextBox());
			this->textBoxBInti = (gcnew System::Windows::Forms::TextBox());
			this->textBoxGInti = (gcnew System::Windows::Forms::TextBox());
			this->textBoxRInti = (gcnew System::Windows::Forms::TextBox());
			this->label59 = (gcnew System::Windows::Forms::Label());
			this->label60 = (gcnew System::Windows::Forms::Label());
			this->label61 = (gcnew System::Windows::Forms::Label());
			this->label62 = (gcnew System::Windows::Forms::Label());
			this->textBoxCircularityInti = (gcnew System::Windows::Forms::TextBox());
			this->label50 = (gcnew System::Windows::Forms::Label());
			this->textBoxGranularityInti = (gcnew System::Windows::Forms::TextBox());
			this->textBoxSolidityInti = (gcnew System::Windows::Forms::TextBox());
			this->textBoxKelilingInti = (gcnew System::Windows::Forms::TextBox());
			this->textBoxLuasInti = (gcnew System::Windows::Forms::TextBox());
			this->label53 = (gcnew System::Windows::Forms::Label());
			this->label54 = (gcnew System::Windows::Forms::Label());
			this->label55 = (gcnew System::Windows::Forms::Label());
			this->label56 = (gcnew System::Windows::Forms::Label());
			this->label57 = (gcnew System::Windows::Forms::Label());
			this->textBoxHomogenityPlasma = (gcnew System::Windows::Forms::TextBox());
			this->label51 = (gcnew System::Windows::Forms::Label());
			this->textBoxContrastPlasma = (gcnew System::Windows::Forms::TextBox());
			this->label52 = (gcnew System::Windows::Forms::Label());
			this->label20 = (gcnew System::Windows::Forms::Label());
			this->textBoxEnergyPlasma = (gcnew System::Windows::Forms::TextBox());
			this->textBoxEntropyPlasma = (gcnew System::Windows::Forms::TextBox());
			this->textBoxBPlasma = (gcnew System::Windows::Forms::TextBox());
			this->textBoxGPlasma = (gcnew System::Windows::Forms::TextBox());
			this->textBoxRPlasma = (gcnew System::Windows::Forms::TextBox());
			this->label27 = (gcnew System::Windows::Forms::Label());
			this->label28 = (gcnew System::Windows::Forms::Label());
			this->label29 = (gcnew System::Windows::Forms::Label());
			this->label30 = (gcnew System::Windows::Forms::Label());
			this->label31 = (gcnew System::Windows::Forms::Label());
			this->textBoxCircularityPlasma = (gcnew System::Windows::Forms::TextBox());
			this->textBoxGranularityPlasma = (gcnew System::Windows::Forms::TextBox());
			this->textBoxSolidityPlasma = (gcnew System::Windows::Forms::TextBox());
			this->textBoxKelilingPlasma = (gcnew System::Windows::Forms::TextBox());
			this->textBoxLuasPlasma = (gcnew System::Windows::Forms::TextBox());
			this->label32 = (gcnew System::Windows::Forms::Label());
			this->label33 = (gcnew System::Windows::Forms::Label());
			this->label34 = (gcnew System::Windows::Forms::Label());
			this->label41 = (gcnew System::Windows::Forms::Label());
			this->label49 = (gcnew System::Windows::Forms::Label());
			this->textBoxLuasNormalisasiInti = (gcnew System::Windows::Forms::TextBox());
			this->label38 = (gcnew System::Windows::Forms::Label());
			this->textBoxEccentricity = (gcnew System::Windows::Forms::TextBox());
			this->textBoxKelilingNormalisasiInti = (gcnew System::Windows::Forms::TextBox());
			this->label35 = (gcnew System::Windows::Forms::Label());
			this->label37 = (gcnew System::Windows::Forms::Label());
			this->textBoxKelilingIntiPlasma = (gcnew System::Windows::Forms::TextBox());
			this->textBoxLuasIntiPlasma = (gcnew System::Windows::Forms::TextBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->label26 = (gcnew System::Windows::Forms::Label());
			this->groupBox3 = (gcnew System::Windows::Forms::GroupBox());
			this->pictureBoxRGB = (gcnew System::Windows::Forms::PictureBox());
			this->pictureBoxSitoplasma = (gcnew System::Windows::Forms::PictureBox());
			this->label48 = (gcnew System::Windows::Forms::Label());
			this->pictureBoxInti = (gcnew System::Windows::Forms::PictureBox());
			this->label47 = (gcnew System::Windows::Forms::Label());
			this->label46 = (gcnew System::Windows::Forms::Label());
			this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->textBoxIntiS = (gcnew System::Windows::Forms::TextBox());
			this->hScrollBarIntiS = (gcnew System::Windows::Forms::HScrollBar());
			this->label9 = (gcnew System::Windows::Forms::Label());
			this->label21 = (gcnew System::Windows::Forms::Label());
			this->textBoxPlasmaV = (gcnew System::Windows::Forms::TextBox());
			this->hScrollBarPlasmaV = (gcnew System::Windows::Forms::HScrollBar());
			this->label23 = (gcnew System::Windows::Forms::Label());
			this->textBoxPlasmaS = (gcnew System::Windows::Forms::TextBox());
			this->hScrollBarPlasmaS = (gcnew System::Windows::Forms::HScrollBar());
			this->label24 = (gcnew System::Windows::Forms::Label());
			this->textBoxPlasmaH = (gcnew System::Windows::Forms::TextBox());
			this->hScrollBarPlasmaH = (gcnew System::Windows::Forms::HScrollBar());
			this->label25 = (gcnew System::Windows::Forms::Label());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->radioButtonConvexArea = (gcnew System::Windows::Forms::RadioButton());
			this->radioButtonConvexHull = (gcnew System::Windows::Forms::RadioButton());
			this->radioButtonKontur = (gcnew System::Windows::Forms::RadioButton());
			this->radioButtonBiner = (gcnew System::Windows::Forms::RadioButton());
			this->folderBrowserDialog = (gcnew System::Windows::Forms::FolderBrowserDialog());
			this->openFileDialogCitra = (gcnew System::Windows::Forms::OpenFileDialog());
			this->statusStrip1->SuspendLayout();
			this->menuStrip1->SuspendLayout();
			this->tabControl1->SuspendLayout();
			this->tabPageLatih->SuspendLayout();
			this->groupBoxMLP->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownBykHdn))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownHdn5))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownHdn4))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownHdn3))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownHdn2))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownOutput))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownHdn1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownInputLayer))->BeginInit();
			this->groupBoxSVM->SuspendLayout();
			this->groupBoxSupervisedData->SuspendLayout();
			this->tabPageEkstraksiFitur->SuspendLayout();
			this->groupBox4->SuspendLayout();
			this->groupBox3->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxRGB))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxSitoplasma))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxInti))->BeginInit();
			this->groupBox2->SuspendLayout();
			this->groupBox1->SuspendLayout();
			this->SuspendLayout();
			// 
			// statusStrip1
			// 
			this->statusStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			this->statusStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				this->toolStripStatusLabel1,
					this->toolStripStatusProgram
			});
			this->statusStrip1->Location = System::Drawing::Point(0, 721);
			this->statusStrip1->Name = L"statusStrip1";
			this->statusStrip1->Padding = System::Windows::Forms::Padding(1, 0, 19, 0);
			this->statusStrip1->Size = System::Drawing::Size(1105, 25);
			this->statusStrip1->TabIndex = 103;
			this->statusStrip1->Text = L"statusStrip1";
			// 
			// toolStripStatusLabel1
			// 
			this->toolStripStatusLabel1->Name = L"toolStripStatusLabel1";
			this->toolStripStatusLabel1->Size = System::Drawing::Size(56, 20);
			this->toolStripStatusLabel1->Text = L"Status: ";
			// 
			// toolStripStatusProgram
			// 
			this->toolStripStatusProgram->Name = L"toolStripStatusProgram";
			this->toolStripStatusProgram->Size = System::Drawing::Size(519, 20);
			this->toolStripStatusProgram->Text = L"Program bisa berjalan. Silakan mulai olah citra dengan memilih menu berkas.";
			// 
			// menuStrip1
			// 
			this->menuStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->berkasToolStripMenuItem });
			this->menuStrip1->Location = System::Drawing::Point(0, 0);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->Padding = System::Windows::Forms::Padding(8, 2, 0, 2);
			this->menuStrip1->Size = System::Drawing::Size(1105, 28);
			this->menuStrip1->TabIndex = 102;
			this->menuStrip1->Text = L"menuStrip1";
			// 
			// berkasToolStripMenuItem
			// 
			this->berkasToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				this->muatCitraEkstraksiToolStripMenuItem,
					this->olahBanyakEkstraksiToolStripMenuItem, this->simpanToolStripMenuItem
			});
			this->berkasToolStripMenuItem->Name = L"berkasToolStripMenuItem";
			this->berkasToolStripMenuItem->Size = System::Drawing::Size(124, 24);
			this->berkasToolStripMenuItem->Text = L"Berkas Ekstraksi";
			// 
			// muatCitraEkstraksiToolStripMenuItem
			// 
			this->muatCitraEkstraksiToolStripMenuItem->Name = L"muatCitraEkstraksiToolStripMenuItem";
			this->muatCitraEkstraksiToolStripMenuItem->Size = System::Drawing::Size(181, 26);
			this->muatCitraEkstraksiToolStripMenuItem->Text = L"Muat Citra";
			this->muatCitraEkstraksiToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::muatCitraEkstraksiToolStripMenuItem_Click);
			// 
			// olahBanyakEkstraksiToolStripMenuItem
			// 
			this->olahBanyakEkstraksiToolStripMenuItem->Name = L"olahBanyakEkstraksiToolStripMenuItem";
			this->olahBanyakEkstraksiToolStripMenuItem->Size = System::Drawing::Size(181, 26);
			this->olahBanyakEkstraksiToolStripMenuItem->Text = L"Olah Banyak";
			this->olahBanyakEkstraksiToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::olahBanyakEkstraksiToolStripMenuItem_Click);
			// 
			// simpanToolStripMenuItem
			// 
			this->simpanToolStripMenuItem->Name = L"simpanToolStripMenuItem";
			this->simpanToolStripMenuItem->Size = System::Drawing::Size(181, 26);
			this->simpanToolStripMenuItem->Text = L"Simpan";
			// 
			// openFileDialogTeks
			// 
			this->openFileDialogTeks->Filter = L"CSV files (*.csv)|*.csv|Text files (*.txt)|*.txt";
			// 
			// tabControl1
			// 
			this->tabControl1->Controls->Add(this->tabPageEkstraksiFitur);
			this->tabControl1->Controls->Add(this->tabPageLatih);
			this->tabControl1->Location = System::Drawing::Point(16, 33);
			this->tabControl1->Margin = System::Windows::Forms::Padding(4);
			this->tabControl1->Name = L"tabControl1";
			this->tabControl1->SelectedIndex = 0;
			this->tabControl1->Size = System::Drawing::Size(1073, 682);
			this->tabControl1->TabIndex = 104;
			// 
			// tabPageLatih
			// 
			this->tabPageLatih->BackColor = System::Drawing::Color::LemonChiffon;
			this->tabPageLatih->Controls->Add(this->labelErrorDataUji);
			this->tabPageLatih->Controls->Add(this->labelErrorDataTes);
			this->tabPageLatih->Controls->Add(this->label40);
			this->tabPageLatih->Controls->Add(this->label39);
			this->tabPageLatih->Controls->Add(this->label36);
			this->tabPageLatih->Controls->Add(this->checkBoxUjiPelatihan);
			this->tabPageLatih->Controls->Add(this->groupBoxMLP);
			this->tabPageLatih->Controls->Add(this->groupBoxSVM);
			this->tabPageLatih->Controls->Add(this->groupBoxSupervisedData);
			this->tabPageLatih->Controls->Add(this->buttonMulaiPelatihan);
			this->tabPageLatih->Controls->Add(this->labelLangkah3);
			this->tabPageLatih->Controls->Add(this->label10);
			this->tabPageLatih->Controls->Add(this->radioButtonDTree);
			this->tabPageLatih->Controls->Add(this->radioButtonBayes);
			this->tabPageLatih->Controls->Add(this->radioButtonKNN);
			this->tabPageLatih->Controls->Add(this->radioButtonMLP);
			this->tabPageLatih->Controls->Add(this->radioButtonLatihSVM);
			this->tabPageLatih->Controls->Add(this->label7);
			this->tabPageLatih->Controls->Add(this->label8);
			this->tabPageLatih->Controls->Add(this->labelDataUji);
			this->tabPageLatih->Controls->Add(this->labelDataLatih);
			this->tabPageLatih->Controls->Add(this->label6);
			this->tabPageLatih->Controls->Add(this->buttonBukaDataUji);
			this->tabPageLatih->Controls->Add(this->buttonBukaDataLatih);
			this->tabPageLatih->Controls->Add(this->label2);
			this->tabPageLatih->Location = System::Drawing::Point(4, 25);
			this->tabPageLatih->Margin = System::Windows::Forms::Padding(4);
			this->tabPageLatih->Name = L"tabPageLatih";
			this->tabPageLatih->Padding = System::Windows::Forms::Padding(4);
			this->tabPageLatih->Size = System::Drawing::Size(1065, 653);
			this->tabPageLatih->TabIndex = 1;
			this->tabPageLatih->Text = L"Latih Program";
			// 
			// labelErrorDataUji
			// 
			this->labelErrorDataUji->AutoSize = true;
			this->labelErrorDataUji->Location = System::Drawing::Point(92, 441);
			this->labelErrorDataUji->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelErrorDataUji->Name = L"labelErrorDataUji";
			this->labelErrorDataUji->Size = System::Drawing::Size(36, 17);
			this->labelErrorDataUji->TabIndex = 57;
			this->labelErrorDataUji->Text = L"0.00";
			// 
			// labelErrorDataTes
			// 
			this->labelErrorDataTes->AutoSize = true;
			this->labelErrorDataTes->Location = System::Drawing::Point(92, 416);
			this->labelErrorDataTes->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelErrorDataTes->Name = L"labelErrorDataTes";
			this->labelErrorDataTes->Size = System::Drawing::Size(36, 17);
			this->labelErrorDataTes->TabIndex = 56;
			this->labelErrorDataTes->Text = L"0.00";
			// 
			// label40
			// 
			this->label40->AutoSize = true;
			this->label40->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label40->Location = System::Drawing::Point(8, 393);
			this->label40->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label40->Name = L"label40";
			this->label40->Size = System::Drawing::Size(93, 17);
			this->label40->TabIndex = 55;
			this->label40->Text = L"Hasil Galat:";
			// 
			// label39
			// 
			this->label39->AutoSize = true;
			this->label39->Location = System::Drawing::Point(12, 441);
			this->label39->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label39->Name = L"label39";
			this->label39->Size = System::Drawing::Size(62, 17);
			this->label39->TabIndex = 54;
			this->label39->Text = L"Data Uji:";
			// 
			// label36
			// 
			this->label36->AutoSize = true;
			this->label36->Location = System::Drawing::Point(12, 416);
			this->label36->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label36->Name = L"label36";
			this->label36->Size = System::Drawing::Size(70, 17);
			this->label36->TabIndex = 53;
			this->label36->Text = L"Data Tes:";
			// 
			// checkBoxUjiPelatihan
			// 
			this->checkBoxUjiPelatihan->AutoSize = true;
			this->checkBoxUjiPelatihan->Location = System::Drawing::Point(12, 191);
			this->checkBoxUjiPelatihan->Margin = System::Windows::Forms::Padding(4);
			this->checkBoxUjiPelatihan->Name = L"checkBoxUjiPelatihan";
			this->checkBoxUjiPelatihan->Size = System::Drawing::Size(109, 21);
			this->checkBoxUjiPelatihan->TabIndex = 52;
			this->checkBoxUjiPelatihan->Text = L"Uji Pelatihan";
			this->checkBoxUjiPelatihan->UseVisualStyleBackColor = true;
			this->checkBoxUjiPelatihan->CheckedChanged += gcnew System::EventHandler(this, &MyForm::checkBoxUjiPelatihan_CheckedChanged);
			// 
			// groupBoxMLP
			// 
			this->groupBoxMLP->Controls->Add(this->labelDWMin);
			this->groupBoxMLP->Controls->Add(this->textBoxDWMin);
			this->groupBoxMLP->Controls->Add(this->labelDW0);
			this->groupBoxMLP->Controls->Add(this->textBoxDW0);
			this->groupBoxMLP->Controls->Add(this->labelMomentum);
			this->groupBoxMLP->Controls->Add(this->textBoxMomentum);
			this->groupBoxMLP->Controls->Add(this->labelWeight);
			this->groupBoxMLP->Controls->Add(this->textBoxWeight);
			this->groupBoxMLP->Controls->Add(this->label44);
			this->groupBoxMLP->Controls->Add(this->textBoxEpsilon);
			this->groupBoxMLP->Controls->Add(this->label45);
			this->groupBoxMLP->Controls->Add(this->textBoxIterasiMLP);
			this->groupBoxMLP->Controls->Add(this->comboBoxActivationFunc);
			this->groupBoxMLP->Controls->Add(this->comboBoxTrainMethod);
			this->groupBoxMLP->Controls->Add(this->label43);
			this->groupBoxMLP->Controls->Add(this->label42);
			this->groupBoxMLP->Controls->Add(this->numericUpDownBykHdn);
			this->groupBoxMLP->Controls->Add(this->labelHdn5);
			this->groupBoxMLP->Controls->Add(this->numericUpDownHdn5);
			this->groupBoxMLP->Controls->Add(this->labelHdn4);
			this->groupBoxMLP->Controls->Add(this->numericUpDownHdn4);
			this->groupBoxMLP->Controls->Add(this->labelHdn3);
			this->groupBoxMLP->Controls->Add(this->numericUpDownHdn3);
			this->groupBoxMLP->Controls->Add(this->labelHdn2);
			this->groupBoxMLP->Controls->Add(this->numericUpDownHdn2);
			this->groupBoxMLP->Controls->Add(this->label22);
			this->groupBoxMLP->Controls->Add(this->numericUpDownOutput);
			this->groupBoxMLP->Controls->Add(this->label19);
			this->groupBoxMLP->Controls->Add(this->numericUpDownHdn1);
			this->groupBoxMLP->Controls->Add(this->label18);
			this->groupBoxMLP->Controls->Add(this->label11);
			this->groupBoxMLP->Controls->Add(this->numericUpDownInputLayer);
			this->groupBoxMLP->Controls->Add(this->label16);
			this->groupBoxMLP->Controls->Add(this->textBoxBeta);
			this->groupBoxMLP->Controls->Add(this->label17);
			this->groupBoxMLP->Controls->Add(this->textBoxAlpha);
			this->groupBoxMLP->Location = System::Drawing::Point(347, 139);
			this->groupBoxMLP->Margin = System::Windows::Forms::Padding(4);
			this->groupBoxMLP->Name = L"groupBoxMLP";
			this->groupBoxMLP->Padding = System::Windows::Forms::Padding(4);
			this->groupBoxMLP->Size = System::Drawing::Size(708, 330);
			this->groupBoxMLP->TabIndex = 51;
			this->groupBoxMLP->TabStop = false;
			this->groupBoxMLP->Text = L"Pengaturan MLP";
			this->groupBoxMLP->Visible = false;
			// 
			// labelDWMin
			// 
			this->labelDWMin->AutoSize = true;
			this->labelDWMin->Location = System::Drawing::Point(20, 283);
			this->labelDWMin->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelDWMin->Name = L"labelDWMin";
			this->labelDWMin->Size = System::Drawing::Size(61, 17);
			this->labelDWMin->TabIndex = 56;
			this->labelDWMin->Text = L"DW Min:";
			this->labelDWMin->Visible = false;
			// 
			// textBoxDWMin
			// 
			this->textBoxDWMin->Location = System::Drawing::Point(89, 278);
			this->textBoxDWMin->Margin = System::Windows::Forms::Padding(4);
			this->textBoxDWMin->Name = L"textBoxDWMin";
			this->textBoxDWMin->Size = System::Drawing::Size(69, 22);
			this->textBoxDWMin->TabIndex = 55;
			this->textBoxDWMin->Text = L"1";
			this->textBoxDWMin->Visible = false;
			// 
			// labelDW0
			// 
			this->labelDW0->AutoSize = true;
			this->labelDW0->Location = System::Drawing::Point(31, 250);
			this->labelDW0->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelDW0->Name = L"labelDW0";
			this->labelDW0->Size = System::Drawing::Size(43, 17);
			this->labelDW0->TabIndex = 54;
			this->labelDW0->Text = L"DW0:";
			this->labelDW0->Visible = false;
			// 
			// textBoxDW0
			// 
			this->textBoxDW0->Location = System::Drawing::Point(89, 245);
			this->textBoxDW0->Margin = System::Windows::Forms::Padding(4);
			this->textBoxDW0->Name = L"textBoxDW0";
			this->textBoxDW0->Size = System::Drawing::Size(69, 22);
			this->textBoxDW0->TabIndex = 53;
			this->textBoxDW0->Text = L"0,1";
			this->textBoxDW0->Visible = false;
			// 
			// labelMomentum
			// 
			this->labelMomentum->AutoSize = true;
			this->labelMomentum->Location = System::Drawing::Point(5, 283);
			this->labelMomentum->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelMomentum->Name = L"labelMomentum";
			this->labelMomentum->Size = System::Drawing::Size(81, 17);
			this->labelMomentum->TabIndex = 52;
			this->labelMomentum->Text = L"Momentum:";
			this->labelMomentum->Visible = false;
			// 
			// textBoxMomentum
			// 
			this->textBoxMomentum->Location = System::Drawing::Point(89, 278);
			this->textBoxMomentum->Margin = System::Windows::Forms::Padding(4);
			this->textBoxMomentum->Name = L"textBoxMomentum";
			this->textBoxMomentum->Size = System::Drawing::Size(69, 22);
			this->textBoxMomentum->TabIndex = 51;
			this->textBoxMomentum->Text = L"0,1";
			this->textBoxMomentum->Visible = false;
			// 
			// labelWeight
			// 
			this->labelWeight->AutoSize = true;
			this->labelWeight->Location = System::Drawing::Point(31, 250);
			this->labelWeight->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelWeight->Name = L"labelWeight";
			this->labelWeight->Size = System::Drawing::Size(56, 17);
			this->labelWeight->TabIndex = 50;
			this->labelWeight->Text = L"Weight:";
			this->labelWeight->Visible = false;
			// 
			// textBoxWeight
			// 
			this->textBoxWeight->Location = System::Drawing::Point(89, 245);
			this->textBoxWeight->Margin = System::Windows::Forms::Padding(4);
			this->textBoxWeight->Name = L"textBoxWeight";
			this->textBoxWeight->Size = System::Drawing::Size(69, 22);
			this->textBoxWeight->TabIndex = 49;
			this->textBoxWeight->Text = L"0,1";
			this->textBoxWeight->Visible = false;
			// 
			// label44
			// 
			this->label44->AutoSize = true;
			this->label44->Location = System::Drawing::Point(31, 217);
			this->label44->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label44->Name = L"label44";
			this->label44->Size = System::Drawing::Size(58, 17);
			this->label44->TabIndex = 48;
			this->label44->Text = L"Epsilon:";
			// 
			// textBoxEpsilon
			// 
			this->textBoxEpsilon->Location = System::Drawing::Point(89, 212);
			this->textBoxEpsilon->Margin = System::Windows::Forms::Padding(4);
			this->textBoxEpsilon->Name = L"textBoxEpsilon";
			this->textBoxEpsilon->Size = System::Drawing::Size(69, 22);
			this->textBoxEpsilon->TabIndex = 47;
			this->textBoxEpsilon->Text = L"0,0001";
			// 
			// label45
			// 
			this->label45->AutoSize = true;
			this->label45->Location = System::Drawing::Point(31, 183);
			this->label45->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label45->Name = L"label45";
			this->label45->Size = System::Drawing::Size(50, 17);
			this->label45->TabIndex = 46;
			this->label45->Text = L"Iterasi:";
			// 
			// textBoxIterasiMLP
			// 
			this->textBoxIterasiMLP->Location = System::Drawing::Point(89, 178);
			this->textBoxIterasiMLP->Margin = System::Windows::Forms::Padding(4);
			this->textBoxIterasiMLP->Name = L"textBoxIterasiMLP";
			this->textBoxIterasiMLP->Size = System::Drawing::Size(69, 22);
			this->textBoxIterasiMLP->TabIndex = 45;
			this->textBoxIterasiMLP->Text = L"100";
			// 
			// comboBoxActivationFunc
			// 
			this->comboBoxActivationFunc->FormattingEnabled = true;
			this->comboBoxActivationFunc->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"Identity", L"Sigmoid", L"Gaussian" });
			this->comboBoxActivationFunc->Location = System::Drawing::Point(237, 144);
			this->comboBoxActivationFunc->Margin = System::Windows::Forms::Padding(4);
			this->comboBoxActivationFunc->Name = L"comboBoxActivationFunc";
			this->comboBoxActivationFunc->Size = System::Drawing::Size(115, 24);
			this->comboBoxActivationFunc->TabIndex = 44;
			this->comboBoxActivationFunc->Text = L"F. Aktivasi";
			this->comboBoxActivationFunc->SelectedIndexChanged += gcnew System::EventHandler(this, &MyForm::comboBoxActivationFunc_SelectedIndexChanged);
			// 
			// comboBoxTrainMethod
			// 
			this->comboBoxTrainMethod->FormattingEnabled = true;
			this->comboBoxTrainMethod->Items->AddRange(gcnew cli::array< System::Object^  >(2) { L"Back Propagation", L"R Propagation" });
			this->comboBoxTrainMethod->Location = System::Drawing::Point(28, 144);
			this->comboBoxTrainMethod->Margin = System::Windows::Forms::Padding(4);
			this->comboBoxTrainMethod->Name = L"comboBoxTrainMethod";
			this->comboBoxTrainMethod->Size = System::Drawing::Size(148, 24);
			this->comboBoxTrainMethod->TabIndex = 43;
			this->comboBoxTrainMethod->Text = L"Metode latih";
			this->comboBoxTrainMethod->SelectedIndexChanged += gcnew System::EventHandler(this, &MyForm::comboBoxTrainMethod_SelectedIndexChanged);
			// 
			// label43
			// 
			this->label43->AutoSize = true;
			this->label43->Location = System::Drawing::Point(8, 124);
			this->label43->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label43->Name = L"label43";
			this->label43->Size = System::Drawing::Size(99, 17);
			this->label43->TabIndex = 42;
			this->label43->Text = L"2. Konfigurasi:";
			// 
			// label42
			// 
			this->label42->AutoSize = true;
			this->label42->Location = System::Drawing::Point(125, 76);
			this->label42->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label42->Name = L"label42";
			this->label42->Size = System::Drawing::Size(108, 17);
			this->label42->TabIndex = 41;
			this->label42->Text = L"Banyak Hidden:";
			// 
			// numericUpDownBykHdn
			// 
			this->numericUpDownBykHdn->Location = System::Drawing::Point(244, 71);
			this->numericUpDownBykHdn->Margin = System::Windows::Forms::Padding(4);
			this->numericUpDownBykHdn->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
			this->numericUpDownBykHdn->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDownBykHdn->Name = L"numericUpDownBykHdn";
			this->numericUpDownBykHdn->Size = System::Drawing::Size(56, 22);
			this->numericUpDownBykHdn->TabIndex = 40;
			this->numericUpDownBykHdn->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDownBykHdn->ValueChanged += gcnew System::EventHandler(this, &MyForm::numericUpDownBykHdn_ValueChanged);
			// 
			// labelHdn5
			// 
			this->labelHdn5->AutoSize = true;
			this->labelHdn5->Location = System::Drawing::Point(595, 42);
			this->labelHdn5->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelHdn5->Name = L"labelHdn5";
			this->labelHdn5->Size = System::Drawing::Size(50, 17);
			this->labelHdn5->TabIndex = 39;
			this->labelHdn5->Text = L"Hdn 5:";
			this->labelHdn5->Visible = false;
			// 
			// numericUpDownHdn5
			// 
			this->numericUpDownHdn5->Location = System::Drawing::Point(648, 39);
			this->numericUpDownHdn5->Margin = System::Windows::Forms::Padding(4);
			this->numericUpDownHdn5->Name = L"numericUpDownHdn5";
			this->numericUpDownHdn5->Size = System::Drawing::Size(56, 22);
			this->numericUpDownHdn5->TabIndex = 38;
			this->numericUpDownHdn5->Visible = false;
			this->numericUpDownHdn5->ValueChanged += gcnew System::EventHandler(this, &MyForm::numericUpDownHdn5_ValueChanged);
			// 
			// labelHdn4
			// 
			this->labelHdn4->AutoSize = true;
			this->labelHdn4->Location = System::Drawing::Point(477, 42);
			this->labelHdn4->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelHdn4->Name = L"labelHdn4";
			this->labelHdn4->Size = System::Drawing::Size(50, 17);
			this->labelHdn4->TabIndex = 37;
			this->labelHdn4->Text = L"Hdn 4:";
			this->labelHdn4->Visible = false;
			// 
			// numericUpDownHdn4
			// 
			this->numericUpDownHdn4->Location = System::Drawing::Point(531, 39);
			this->numericUpDownHdn4->Margin = System::Windows::Forms::Padding(4);
			this->numericUpDownHdn4->Name = L"numericUpDownHdn4";
			this->numericUpDownHdn4->Size = System::Drawing::Size(56, 22);
			this->numericUpDownHdn4->TabIndex = 36;
			this->numericUpDownHdn4->Visible = false;
			this->numericUpDownHdn4->ValueChanged += gcnew System::EventHandler(this, &MyForm::numericUpDownHdn4_ValueChanged);
			// 
			// labelHdn3
			// 
			this->labelHdn3->AutoSize = true;
			this->labelHdn3->Location = System::Drawing::Point(360, 42);
			this->labelHdn3->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelHdn3->Name = L"labelHdn3";
			this->labelHdn3->Size = System::Drawing::Size(50, 17);
			this->labelHdn3->TabIndex = 35;
			this->labelHdn3->Text = L"Hdn 3:";
			this->labelHdn3->Visible = false;
			// 
			// numericUpDownHdn3
			// 
			this->numericUpDownHdn3->Location = System::Drawing::Point(413, 39);
			this->numericUpDownHdn3->Margin = System::Windows::Forms::Padding(4);
			this->numericUpDownHdn3->Name = L"numericUpDownHdn3";
			this->numericUpDownHdn3->Size = System::Drawing::Size(56, 22);
			this->numericUpDownHdn3->TabIndex = 34;
			this->numericUpDownHdn3->Visible = false;
			this->numericUpDownHdn3->ValueChanged += gcnew System::EventHandler(this, &MyForm::numericUpDownHdn3_ValueChanged);
			// 
			// labelHdn2
			// 
			this->labelHdn2->AutoSize = true;
			this->labelHdn2->Location = System::Drawing::Point(243, 42);
			this->labelHdn2->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelHdn2->Name = L"labelHdn2";
			this->labelHdn2->Size = System::Drawing::Size(50, 17);
			this->labelHdn2->TabIndex = 33;
			this->labelHdn2->Text = L"Hdn 2:";
			this->labelHdn2->Visible = false;
			// 
			// numericUpDownHdn2
			// 
			this->numericUpDownHdn2->Location = System::Drawing::Point(296, 39);
			this->numericUpDownHdn2->Margin = System::Windows::Forms::Padding(4);
			this->numericUpDownHdn2->Name = L"numericUpDownHdn2";
			this->numericUpDownHdn2->Size = System::Drawing::Size(56, 22);
			this->numericUpDownHdn2->TabIndex = 32;
			this->numericUpDownHdn2->Visible = false;
			this->numericUpDownHdn2->ValueChanged += gcnew System::EventHandler(this, &MyForm::numericUpDownHdn2_ValueChanged);
			// 
			// label22
			// 
			this->label22->AutoSize = true;
			this->label22->Location = System::Drawing::Point(1, 76);
			this->label22->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label22->Name = L"label22";
			this->label22->Size = System::Drawing::Size(55, 17);
			this->label22->TabIndex = 31;
			this->label22->Text = L"Output:";
			// 
			// numericUpDownOutput
			// 
			this->numericUpDownOutput->Location = System::Drawing::Point(61, 74);
			this->numericUpDownOutput->Margin = System::Windows::Forms::Padding(4);
			this->numericUpDownOutput->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDownOutput->Name = L"numericUpDownOutput";
			this->numericUpDownOutput->Size = System::Drawing::Size(56, 22);
			this->numericUpDownOutput->TabIndex = 30;
			this->numericUpDownOutput->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDownOutput->ValueChanged += gcnew System::EventHandler(this, &MyForm::numericUpDownOutput_ValueChanged);
			// 
			// label19
			// 
			this->label19->AutoSize = true;
			this->label19->Location = System::Drawing::Point(125, 42);
			this->label19->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label19->Name = L"label19";
			this->label19->Size = System::Drawing::Size(50, 17);
			this->label19->TabIndex = 29;
			this->label19->Text = L"Hdn 1:";
			// 
			// numericUpDownHdn1
			// 
			this->numericUpDownHdn1->Location = System::Drawing::Point(179, 39);
			this->numericUpDownHdn1->Margin = System::Windows::Forms::Padding(4);
			this->numericUpDownHdn1->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDownHdn1->Name = L"numericUpDownHdn1";
			this->numericUpDownHdn1->Size = System::Drawing::Size(56, 22);
			this->numericUpDownHdn1->TabIndex = 28;
			this->numericUpDownHdn1->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDownHdn1->ValueChanged += gcnew System::EventHandler(this, &MyForm::numericUpDownHdn1_ValueChanged);
			// 
			// label18
			// 
			this->label18->AutoSize = true;
			this->label18->Location = System::Drawing::Point(8, 42);
			this->label18->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label18->Name = L"label18";
			this->label18->Size = System::Drawing::Size(43, 17);
			this->label18->TabIndex = 27;
			this->label18->Text = L"Input:";
			// 
			// label11
			// 
			this->label11->AutoSize = true;
			this->label11->Location = System::Drawing::Point(8, 20);
			this->label11->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label11->Name = L"label11";
			this->label11->Size = System::Drawing::Size(64, 17);
			this->label11->TabIndex = 26;
			this->label11->Text = L"1. Layer:";
			// 
			// numericUpDownInputLayer
			// 
			this->numericUpDownInputLayer->Location = System::Drawing::Point(61, 39);
			this->numericUpDownInputLayer->Margin = System::Windows::Forms::Padding(4);
			this->numericUpDownInputLayer->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDownInputLayer->Name = L"numericUpDownInputLayer";
			this->numericUpDownInputLayer->Size = System::Drawing::Size(56, 22);
			this->numericUpDownInputLayer->TabIndex = 25;
			this->numericUpDownInputLayer->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDownInputLayer->ValueChanged += gcnew System::EventHandler(this, &MyForm::numericUpDownInputLayer_ValueChanged);
			// 
			// label16
			// 
			this->label16->AutoSize = true;
			this->label16->Location = System::Drawing::Point(236, 215);
			this->label16->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label16->Name = L"label16";
			this->label16->Size = System::Drawing::Size(41, 17);
			this->label16->TabIndex = 24;
			this->label16->Text = L"Beta:";
			// 
			// textBoxBeta
			// 
			this->textBoxBeta->Location = System::Drawing::Point(281, 210);
			this->textBoxBeta->Margin = System::Windows::Forms::Padding(4);
			this->textBoxBeta->Name = L"textBoxBeta";
			this->textBoxBeta->Size = System::Drawing::Size(69, 22);
			this->textBoxBeta->TabIndex = 23;
			this->textBoxBeta->Text = L"0,00";
			// 
			// label17
			// 
			this->label17->AutoSize = true;
			this->label17->Location = System::Drawing::Point(236, 182);
			this->label17->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label17->Name = L"label17";
			this->label17->Size = System::Drawing::Size(36, 17);
			this->label17->TabIndex = 22;
			this->label17->Text = L"Alfa:";
			// 
			// textBoxAlpha
			// 
			this->textBoxAlpha->Location = System::Drawing::Point(281, 177);
			this->textBoxAlpha->Margin = System::Windows::Forms::Padding(4);
			this->textBoxAlpha->Name = L"textBoxAlpha";
			this->textBoxAlpha->Size = System::Drawing::Size(69, 22);
			this->textBoxAlpha->TabIndex = 20;
			this->textBoxAlpha->Text = L"0,00";
			// 
			// groupBoxSVM
			// 
			this->groupBoxSVM->Controls->Add(this->label14);
			this->groupBoxSVM->Controls->Add(this->textBox3);
			this->groupBoxSVM->Controls->Add(this->label15);
			this->groupBoxSVM->Controls->Add(this->textBox4);
			this->groupBoxSVM->Location = System::Drawing::Point(347, 476);
			this->groupBoxSVM->Margin = System::Windows::Forms::Padding(4);
			this->groupBoxSVM->Name = L"groupBoxSVM";
			this->groupBoxSVM->Padding = System::Windows::Forms::Padding(4);
			this->groupBoxSVM->Size = System::Drawing::Size(353, 111);
			this->groupBoxSVM->TabIndex = 50;
			this->groupBoxSVM->TabStop = false;
			this->groupBoxSVM->Text = L"Pengaturan SVM";
			this->groupBoxSVM->Visible = false;
			// 
			// label14
			// 
			this->label14->AutoSize = true;
			this->label14->Location = System::Drawing::Point(8, 65);
			this->label14->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label14->Name = L"label14";
			this->label14->Size = System::Drawing::Size(108, 17);
			this->label14->TabIndex = 24;
			this->label14->Text = L"Variabel output:";
			// 
			// textBox3
			// 
			this->textBox3->Location = System::Drawing::Point(128, 62);
			this->textBox3->Margin = System::Windows::Forms::Padding(4);
			this->textBox3->Name = L"textBox3";
			this->textBox3->Size = System::Drawing::Size(132, 22);
			this->textBox3->TabIndex = 23;
			// 
			// label15
			// 
			this->label15->AutoSize = true;
			this->label15->Location = System::Drawing::Point(8, 32);
			this->label15->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label15->Name = L"label15";
			this->label15->Size = System::Drawing::Size(99, 17);
			this->label15->TabIndex = 22;
			this->label15->Text = L"Variabel input:";
			// 
			// textBox4
			// 
			this->textBox4->Location = System::Drawing::Point(128, 28);
			this->textBox4->Margin = System::Windows::Forms::Padding(4);
			this->textBox4->Name = L"textBox4";
			this->textBox4->Size = System::Drawing::Size(132, 22);
			this->textBox4->TabIndex = 20;
			// 
			// groupBoxSupervisedData
			// 
			this->groupBoxSupervisedData->Controls->Add(this->label3);
			this->groupBoxSupervisedData->Controls->Add(this->label1);
			this->groupBoxSupervisedData->Controls->Add(this->label13);
			this->groupBoxSupervisedData->Controls->Add(this->textBoxBanyakOutput);
			this->groupBoxSupervisedData->Controls->Add(this->label12);
			this->groupBoxSupervisedData->Controls->Add(this->textBoxBanyakInput);
			this->groupBoxSupervisedData->Location = System::Drawing::Point(347, 17);
			this->groupBoxSupervisedData->Margin = System::Windows::Forms::Padding(4);
			this->groupBoxSupervisedData->Name = L"groupBoxSupervisedData";
			this->groupBoxSupervisedData->Padding = System::Windows::Forms::Padding(4);
			this->groupBoxSupervisedData->Size = System::Drawing::Size(704, 111);
			this->groupBoxSupervisedData->TabIndex = 49;
			this->groupBoxSupervisedData->TabStop = false;
			this->groupBoxSupervisedData->Text = L"Pengaturan Langkah 1";
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(187, 32);
			this->label3->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(403, 17);
			this->label3->TabIndex = 59;
			this->label3->Text = L"Banyaknya variabel input yang terdapat pada data latih dan uji";
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(187, 65);
			this->label1->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(474, 17);
			this->label1->TabIndex = 58;
			this->label1->Text = L"Nilai -1 menunjukkan semua variabel setelah input adalah variabel output.";
			// 
			// label13
			// 
			this->label13->AutoSize = true;
			this->label13->Location = System::Drawing::Point(8, 65);
			this->label13->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label13->Name = L"label13";
			this->label13->Size = System::Drawing::Size(108, 17);
			this->label13->TabIndex = 24;
			this->label13->Text = L"Variabel output:";
			// 
			// textBoxBanyakOutput
			// 
			this->textBoxBanyakOutput->Location = System::Drawing::Point(128, 62);
			this->textBoxBanyakOutput->Margin = System::Windows::Forms::Padding(4);
			this->textBoxBanyakOutput->Name = L"textBoxBanyakOutput";
			this->textBoxBanyakOutput->Size = System::Drawing::Size(49, 22);
			this->textBoxBanyakOutput->TabIndex = 23;
			this->textBoxBanyakOutput->Text = L"-1";
			// 
			// label12
			// 
			this->label12->AutoSize = true;
			this->label12->Location = System::Drawing::Point(8, 32);
			this->label12->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label12->Name = L"label12";
			this->label12->Size = System::Drawing::Size(99, 17);
			this->label12->TabIndex = 22;
			this->label12->Text = L"Variabel input:";
			// 
			// textBoxBanyakInput
			// 
			this->textBoxBanyakInput->Location = System::Drawing::Point(128, 28);
			this->textBoxBanyakInput->Margin = System::Windows::Forms::Padding(4);
			this->textBoxBanyakInput->Name = L"textBoxBanyakInput";
			this->textBoxBanyakInput->Size = System::Drawing::Size(49, 22);
			this->textBoxBanyakInput->TabIndex = 20;
			this->textBoxBanyakInput->Text = L"3";
			// 
			// buttonMulaiPelatihan
			// 
			this->buttonMulaiPelatihan->Location = System::Drawing::Point(12, 348);
			this->buttonMulaiPelatihan->Margin = System::Windows::Forms::Padding(4);
			this->buttonMulaiPelatihan->Name = L"buttonMulaiPelatihan";
			this->buttonMulaiPelatihan->Size = System::Drawing::Size(137, 28);
			this->buttonMulaiPelatihan->TabIndex = 48;
			this->buttonMulaiPelatihan->Text = L"Mulai Latih";
			this->buttonMulaiPelatihan->UseVisualStyleBackColor = true;
			this->buttonMulaiPelatihan->Click += gcnew System::EventHandler(this, &MyForm::buttonMulaiPelatihan_Click);
			// 
			// labelLangkah3
			// 
			this->labelLangkah3->AutoSize = true;
			this->labelLangkah3->Location = System::Drawing::Point(8, 322);
			this->labelLangkah3->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelLangkah3->Name = L"labelLangkah3";
			this->labelLangkah3->Size = System::Drawing::Size(131, 17);
			this->labelLangkah3->TabIndex = 47;
			this->labelLangkah3->Text = L"Jalankan pelatihan.";
			// 
			// label10
			// 
			this->label10->AutoSize = true;
			this->label10->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label10->Location = System::Drawing::Point(8, 306);
			this->label10->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label10->Name = L"label10";
			this->label10->Size = System::Drawing::Size(89, 17);
			this->label10->TabIndex = 46;
			this->label10->Text = L"Langkah 3:";
			// 
			// radioButtonDTree
			// 
			this->radioButtonDTree->AutoSize = true;
			this->radioButtonDTree->Location = System::Drawing::Point(133, 81);
			this->radioButtonDTree->Margin = System::Windows::Forms::Padding(4);
			this->radioButtonDTree->Name = L"radioButtonDTree";
			this->radioButtonDTree->Size = System::Drawing::Size(77, 21);
			this->radioButtonDTree->TabIndex = 45;
			this->radioButtonDTree->Text = L"D. Tree";
			this->radioButtonDTree->UseVisualStyleBackColor = true;
			this->radioButtonDTree->Click += gcnew System::EventHandler(this, &MyForm::radioButtonDTree_Click);
			// 
			// radioButtonBayes
			// 
			this->radioButtonBayes->AutoSize = true;
			this->radioButtonBayes->Location = System::Drawing::Point(12, 81);
			this->radioButtonBayes->Margin = System::Windows::Forms::Padding(4);
			this->radioButtonBayes->Name = L"radioButtonBayes";
			this->radioButtonBayes->Size = System::Drawing::Size(68, 21);
			this->radioButtonBayes->TabIndex = 44;
			this->radioButtonBayes->Text = L"Bayes";
			this->radioButtonBayes->UseVisualStyleBackColor = true;
			this->radioButtonBayes->Click += gcnew System::EventHandler(this, &MyForm::radioButtonBayes_Click);
			// 
			// radioButtonKNN
			// 
			this->radioButtonKNN->AutoSize = true;
			this->radioButtonKNN->Location = System::Drawing::Point(255, 53);
			this->radioButtonKNN->Margin = System::Windows::Forms::Padding(4);
			this->radioButtonKNN->Name = L"radioButtonKNN";
			this->radioButtonKNN->Size = System::Drawing::Size(58, 21);
			this->radioButtonKNN->TabIndex = 43;
			this->radioButtonKNN->Text = L"KNN";
			this->radioButtonKNN->UseVisualStyleBackColor = true;
			this->radioButtonKNN->Click += gcnew System::EventHandler(this, &MyForm::radioButtonKNN_Click);
			// 
			// radioButtonMLP
			// 
			this->radioButtonMLP->AutoSize = true;
			this->radioButtonMLP->Checked = true;
			this->radioButtonMLP->Location = System::Drawing::Point(12, 53);
			this->radioButtonMLP->Margin = System::Windows::Forms::Padding(4);
			this->radioButtonMLP->Name = L"radioButtonMLP";
			this->radioButtonMLP->Size = System::Drawing::Size(57, 21);
			this->radioButtonMLP->TabIndex = 42;
			this->radioButtonMLP->TabStop = true;
			this->radioButtonMLP->Text = L"MLP";
			this->radioButtonMLP->UseVisualStyleBackColor = true;
			this->radioButtonMLP->Click += gcnew System::EventHandler(this, &MyForm::radioButtonMLP_Click);
			// 
			// radioButtonLatihSVM
			// 
			this->radioButtonLatihSVM->AutoSize = true;
			this->radioButtonLatihSVM->Location = System::Drawing::Point(133, 53);
			this->radioButtonLatihSVM->Margin = System::Windows::Forms::Padding(4);
			this->radioButtonLatihSVM->Name = L"radioButtonLatihSVM";
			this->radioButtonLatihSVM->Size = System::Drawing::Size(58, 21);
			this->radioButtonLatihSVM->TabIndex = 41;
			this->radioButtonLatihSVM->Text = L"SVM";
			this->radioButtonLatihSVM->UseVisualStyleBackColor = true;
			this->radioButtonLatihSVM->Click += gcnew System::EventHandler(this, &MyForm::radioButtonLatihSVM_Click);
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(8, 33);
			this->label7->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(176, 17);
			this->label7->TabIndex = 40;
			this->label7->Text = L"Pilih metode latih program.";
			// 
			// label8
			// 
			this->label8->AutoSize = true;
			this->label8->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label8->Location = System::Drawing::Point(8, 17);
			this->label8->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label8->Name = L"label8";
			this->label8->Size = System::Drawing::Size(89, 17);
			this->label8->TabIndex = 39;
			this->label8->Text = L"Langkah 1:";
			// 
			// labelDataUji
			// 
			this->labelDataUji->AutoSize = true;
			this->labelDataUji->Location = System::Drawing::Point(157, 257);
			this->labelDataUji->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelDataUji->Name = L"labelDataUji";
			this->labelDataUji->Size = System::Drawing::Size(126, 17);
			this->labelDataUji->TabIndex = 38;
			this->labelDataUji->Text = L"Belum ada berkas.";
			// 
			// labelDataLatih
			// 
			this->labelDataLatih->AutoSize = true;
			this->labelDataLatih->Location = System::Drawing::Point(157, 222);
			this->labelDataLatih->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->labelDataLatih->Name = L"labelDataLatih";
			this->labelDataLatih->Size = System::Drawing::Size(126, 17);
			this->labelDataLatih->TabIndex = 37;
			this->labelDataLatih->Text = L"Belum ada berkas.";
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Location = System::Drawing::Point(8, 155);
			this->label6->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(330, 34);
			this->label6->TabIndex = 36;
			this->label6->Text = L"Buka berkas yang dibutuhkan untuk latih program. \r\nFormat CSV dengan pemisah \';\'";
			// 
			// buttonBukaDataUji
			// 
			this->buttonBukaDataUji->Location = System::Drawing::Point(12, 251);
			this->buttonBukaDataUji->Margin = System::Windows::Forms::Padding(4);
			this->buttonBukaDataUji->Name = L"buttonBukaDataUji";
			this->buttonBukaDataUji->Size = System::Drawing::Size(137, 28);
			this->buttonBukaDataUji->TabIndex = 35;
			this->buttonBukaDataUji->Text = L"Data uji";
			this->buttonBukaDataUji->UseVisualStyleBackColor = true;
			this->buttonBukaDataUji->Click += gcnew System::EventHandler(this, &MyForm::buttonBukaDataUji_Click);
			// 
			// buttonBukaDataLatih
			// 
			this->buttonBukaDataLatih->Location = System::Drawing::Point(12, 215);
			this->buttonBukaDataLatih->Margin = System::Windows::Forms::Padding(4);
			this->buttonBukaDataLatih->Name = L"buttonBukaDataLatih";
			this->buttonBukaDataLatih->Size = System::Drawing::Size(137, 28);
			this->buttonBukaDataLatih->TabIndex = 34;
			this->buttonBukaDataLatih->Text = L"Data latih";
			this->buttonBukaDataLatih->UseVisualStyleBackColor = true;
			this->buttonBukaDataLatih->Click += gcnew System::EventHandler(this, &MyForm::buttonBukaDataLatih_Click);
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label2->Location = System::Drawing::Point(8, 139);
			this->label2->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(89, 17);
			this->label2->TabIndex = 33;
			this->label2->Text = L"Langkah 2:";
			// 
			// tabPageEkstraksiFitur
			// 
			this->tabPageEkstraksiFitur->BackColor = System::Drawing::Color::Moccasin;
			this->tabPageEkstraksiFitur->Controls->Add(this->groupBox4);
			this->tabPageEkstraksiFitur->Controls->Add(this->groupBox3);
			this->tabPageEkstraksiFitur->Controls->Add(this->groupBox2);
			this->tabPageEkstraksiFitur->Controls->Add(this->groupBox1);
			this->tabPageEkstraksiFitur->Location = System::Drawing::Point(4, 25);
			this->tabPageEkstraksiFitur->Margin = System::Windows::Forms::Padding(4);
			this->tabPageEkstraksiFitur->Name = L"tabPageEkstraksiFitur";
			this->tabPageEkstraksiFitur->Padding = System::Windows::Forms::Padding(4);
			this->tabPageEkstraksiFitur->Size = System::Drawing::Size(1065, 653);
			this->tabPageEkstraksiFitur->TabIndex = 0;
			this->tabPageEkstraksiFitur->Text = L"Ekstraksi Fitur";
			// 
			// groupBox4
			// 
			this->groupBox4->Controls->Add(this->textBoxHomogenityInti);
			this->groupBox4->Controls->Add(this->label63);
			this->groupBox4->Controls->Add(this->textBoxContrastInti);
			this->groupBox4->Controls->Add(this->label64);
			this->groupBox4->Controls->Add(this->textBoxEnergyInti);
			this->groupBox4->Controls->Add(this->label58);
			this->groupBox4->Controls->Add(this->textBoxEntropyInti);
			this->groupBox4->Controls->Add(this->textBoxBInti);
			this->groupBox4->Controls->Add(this->textBoxGInti);
			this->groupBox4->Controls->Add(this->textBoxRInti);
			this->groupBox4->Controls->Add(this->label59);
			this->groupBox4->Controls->Add(this->label60);
			this->groupBox4->Controls->Add(this->label61);
			this->groupBox4->Controls->Add(this->label62);
			this->groupBox4->Controls->Add(this->textBoxCircularityInti);
			this->groupBox4->Controls->Add(this->label50);
			this->groupBox4->Controls->Add(this->textBoxGranularityInti);
			this->groupBox4->Controls->Add(this->textBoxSolidityInti);
			this->groupBox4->Controls->Add(this->textBoxKelilingInti);
			this->groupBox4->Controls->Add(this->textBoxLuasInti);
			this->groupBox4->Controls->Add(this->label53);
			this->groupBox4->Controls->Add(this->label54);
			this->groupBox4->Controls->Add(this->label55);
			this->groupBox4->Controls->Add(this->label56);
			this->groupBox4->Controls->Add(this->label57);
			this->groupBox4->Controls->Add(this->textBoxHomogenityPlasma);
			this->groupBox4->Controls->Add(this->label51);
			this->groupBox4->Controls->Add(this->textBoxContrastPlasma);
			this->groupBox4->Controls->Add(this->label52);
			this->groupBox4->Controls->Add(this->label20);
			this->groupBox4->Controls->Add(this->textBoxEnergyPlasma);
			this->groupBox4->Controls->Add(this->textBoxEntropyPlasma);
			this->groupBox4->Controls->Add(this->textBoxBPlasma);
			this->groupBox4->Controls->Add(this->textBoxGPlasma);
			this->groupBox4->Controls->Add(this->textBoxRPlasma);
			this->groupBox4->Controls->Add(this->label27);
			this->groupBox4->Controls->Add(this->label28);
			this->groupBox4->Controls->Add(this->label29);
			this->groupBox4->Controls->Add(this->label30);
			this->groupBox4->Controls->Add(this->label31);
			this->groupBox4->Controls->Add(this->textBoxCircularityPlasma);
			this->groupBox4->Controls->Add(this->textBoxGranularityPlasma);
			this->groupBox4->Controls->Add(this->textBoxSolidityPlasma);
			this->groupBox4->Controls->Add(this->textBoxKelilingPlasma);
			this->groupBox4->Controls->Add(this->textBoxLuasPlasma);
			this->groupBox4->Controls->Add(this->label32);
			this->groupBox4->Controls->Add(this->label33);
			this->groupBox4->Controls->Add(this->label34);
			this->groupBox4->Controls->Add(this->label41);
			this->groupBox4->Controls->Add(this->label49);
			this->groupBox4->Controls->Add(this->textBoxLuasNormalisasiInti);
			this->groupBox4->Controls->Add(this->label38);
			this->groupBox4->Controls->Add(this->textBoxEccentricity);
			this->groupBox4->Controls->Add(this->textBoxKelilingNormalisasiInti);
			this->groupBox4->Controls->Add(this->label35);
			this->groupBox4->Controls->Add(this->label37);
			this->groupBox4->Controls->Add(this->textBoxKelilingIntiPlasma);
			this->groupBox4->Controls->Add(this->textBoxLuasIntiPlasma);
			this->groupBox4->Controls->Add(this->label4);
			this->groupBox4->Controls->Add(this->label26);
			this->groupBox4->Location = System::Drawing::Point(477, 20);
			this->groupBox4->Name = L"groupBox4";
			this->groupBox4->Size = System::Drawing::Size(570, 610);
			this->groupBox4->TabIndex = 155;
			this->groupBox4->TabStop = false;
			this->groupBox4->Text = L"Fitur Citra:";
			// 
			// textBoxHomogenityInti
			// 
			this->textBoxHomogenityInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxHomogenityInti->Location = System::Drawing::Point(353, 546);
			this->textBoxHomogenityInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxHomogenityInti->Name = L"textBoxHomogenityInti";
			this->textBoxHomogenityInti->ReadOnly = true;
			this->textBoxHomogenityInti->Size = System::Drawing::Size(95, 22);
			this->textBoxHomogenityInti->TabIndex = 210;
			this->textBoxHomogenityInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label63
			// 
			this->label63->AutoSize = true;
			this->label63->Location = System::Drawing::Point(252, 550);
			this->label63->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label63->Name = L"label63";
			this->label63->Size = System::Drawing::Size(87, 17);
			this->label63->TabIndex = 209;
			this->label63->Text = L"Homogenity:";
			// 
			// textBoxContrastInti
			// 
			this->textBoxContrastInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxContrastInti->Location = System::Drawing::Point(353, 518);
			this->textBoxContrastInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxContrastInti->Name = L"textBoxContrastInti";
			this->textBoxContrastInti->ReadOnly = true;
			this->textBoxContrastInti->Size = System::Drawing::Size(95, 22);
			this->textBoxContrastInti->TabIndex = 208;
			this->textBoxContrastInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label64
			// 
			this->label64->AutoSize = true;
			this->label64->Location = System::Drawing::Point(252, 521);
			this->label64->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label64->Name = L"label64";
			this->label64->Size = System::Drawing::Size(65, 17);
			this->label64->TabIndex = 207;
			this->label64->Text = L"Contrast:";
			// 
			// textBoxEnergyInti
			// 
			this->textBoxEnergyInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxEnergyInti->Location = System::Drawing::Point(353, 487);
			this->textBoxEnergyInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxEnergyInti->Name = L"textBoxEnergyInti";
			this->textBoxEnergyInti->ReadOnly = true;
			this->textBoxEnergyInti->Size = System::Drawing::Size(95, 22);
			this->textBoxEnergyInti->TabIndex = 206;
			this->textBoxEnergyInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label58
			// 
			this->label58->AutoSize = true;
			this->label58->Location = System::Drawing::Point(252, 491);
			this->label58->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label58->Name = L"label58";
			this->label58->Size = System::Drawing::Size(57, 17);
			this->label58->TabIndex = 205;
			this->label58->Text = L"Energy:";
			// 
			// textBoxEntropyInti
			// 
			this->textBoxEntropyInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxEntropyInti->Location = System::Drawing::Point(353, 459);
			this->textBoxEntropyInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxEntropyInti->Name = L"textBoxEntropyInti";
			this->textBoxEntropyInti->ReadOnly = true;
			this->textBoxEntropyInti->Size = System::Drawing::Size(95, 22);
			this->textBoxEntropyInti->TabIndex = 204;
			this->textBoxEntropyInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxBInti
			// 
			this->textBoxBInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxBInti->Location = System::Drawing::Point(353, 430);
			this->textBoxBInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxBInti->Name = L"textBoxBInti";
			this->textBoxBInti->ReadOnly = true;
			this->textBoxBInti->Size = System::Drawing::Size(95, 22);
			this->textBoxBInti->TabIndex = 203;
			this->textBoxBInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxGInti
			// 
			this->textBoxGInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxGInti->Location = System::Drawing::Point(353, 402);
			this->textBoxGInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxGInti->Name = L"textBoxGInti";
			this->textBoxGInti->ReadOnly = true;
			this->textBoxGInti->Size = System::Drawing::Size(95, 22);
			this->textBoxGInti->TabIndex = 202;
			this->textBoxGInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxRInti
			// 
			this->textBoxRInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxRInti->Location = System::Drawing::Point(353, 374);
			this->textBoxRInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxRInti->Name = L"textBoxRInti";
			this->textBoxRInti->ReadOnly = true;
			this->textBoxRInti->Size = System::Drawing::Size(95, 22);
			this->textBoxRInti->TabIndex = 201;
			this->textBoxRInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label59
			// 
			this->label59->AutoSize = true;
			this->label59->Location = System::Drawing::Point(252, 462);
			this->label59->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label59->Name = L"label59";
			this->label59->Size = System::Drawing::Size(61, 17);
			this->label59->TabIndex = 200;
			this->label59->Text = L"Entropy:";
			// 
			// label60
			// 
			this->label60->AutoSize = true;
			this->label60->Location = System::Drawing::Point(252, 434);
			this->label60->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label60->Name = L"label60";
			this->label60->Size = System::Drawing::Size(61, 17);
			this->label60->TabIndex = 199;
			this->label60->Text = L"Kanal B:";
			// 
			// label61
			// 
			this->label61->AutoSize = true;
			this->label61->Location = System::Drawing::Point(252, 406);
			this->label61->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label61->Name = L"label61";
			this->label61->Size = System::Drawing::Size(63, 17);
			this->label61->TabIndex = 198;
			this->label61->Text = L"Kanal G:";
			// 
			// label62
			// 
			this->label62->AutoSize = true;
			this->label62->Location = System::Drawing::Point(252, 378);
			this->label62->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label62->Name = L"label62";
			this->label62->Size = System::Drawing::Size(62, 17);
			this->label62->TabIndex = 197;
			this->label62->Text = L"Kanal R:";
			// 
			// textBoxCircularityInti
			// 
			this->textBoxCircularityInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxCircularityInti->Location = System::Drawing::Point(354, 346);
			this->textBoxCircularityInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxCircularityInti->Name = L"textBoxCircularityInti";
			this->textBoxCircularityInti->ReadOnly = true;
			this->textBoxCircularityInti->Size = System::Drawing::Size(95, 22);
			this->textBoxCircularityInti->TabIndex = 196;
			this->textBoxCircularityInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label50
			// 
			this->label50->AutoSize = true;
			this->label50->Location = System::Drawing::Point(252, 349);
			this->label50->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label50->Name = L"label50";
			this->label50->Size = System::Drawing::Size(74, 17);
			this->label50->TabIndex = 195;
			this->label50->Text = L"Circularity:";
			// 
			// textBoxGranularityInti
			// 
			this->textBoxGranularityInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxGranularityInti->Location = System::Drawing::Point(354, 317);
			this->textBoxGranularityInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxGranularityInti->Name = L"textBoxGranularityInti";
			this->textBoxGranularityInti->ReadOnly = true;
			this->textBoxGranularityInti->Size = System::Drawing::Size(95, 22);
			this->textBoxGranularityInti->TabIndex = 194;
			this->textBoxGranularityInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxSolidityInti
			// 
			this->textBoxSolidityInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxSolidityInti->Location = System::Drawing::Point(354, 289);
			this->textBoxSolidityInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxSolidityInti->Name = L"textBoxSolidityInti";
			this->textBoxSolidityInti->ReadOnly = true;
			this->textBoxSolidityInti->Size = System::Drawing::Size(95, 22);
			this->textBoxSolidityInti->TabIndex = 193;
			this->textBoxSolidityInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxKelilingInti
			// 
			this->textBoxKelilingInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxKelilingInti->Location = System::Drawing::Point(354, 261);
			this->textBoxKelilingInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxKelilingInti->Name = L"textBoxKelilingInti";
			this->textBoxKelilingInti->ReadOnly = true;
			this->textBoxKelilingInti->Size = System::Drawing::Size(95, 22);
			this->textBoxKelilingInti->TabIndex = 192;
			this->textBoxKelilingInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxLuasInti
			// 
			this->textBoxLuasInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxLuasInti->Location = System::Drawing::Point(354, 233);
			this->textBoxLuasInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxLuasInti->Name = L"textBoxLuasInti";
			this->textBoxLuasInti->ReadOnly = true;
			this->textBoxLuasInti->Size = System::Drawing::Size(95, 22);
			this->textBoxLuasInti->TabIndex = 191;
			this->textBoxLuasInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label53
			// 
			this->label53->AutoSize = true;
			this->label53->Location = System::Drawing::Point(252, 321);
			this->label53->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label53->Name = L"label53";
			this->label53->Size = System::Drawing::Size(82, 17);
			this->label53->TabIndex = 190;
			this->label53->Text = L"Granularity:";
			// 
			// label54
			// 
			this->label54->AutoSize = true;
			this->label54->Location = System::Drawing::Point(252, 293);
			this->label54->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label54->Name = L"label54";
			this->label54->Size = System::Drawing::Size(57, 17);
			this->label54->TabIndex = 189;
			this->label54->Text = L"Solidity:";
			// 
			// label55
			// 
			this->label55->AutoSize = true;
			this->label55->Location = System::Drawing::Point(252, 265);
			this->label55->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label55->Name = L"label55";
			this->label55->Size = System::Drawing::Size(57, 17);
			this->label55->TabIndex = 188;
			this->label55->Text = L"Keliling:";
			// 
			// label56
			// 
			this->label56->AutoSize = true;
			this->label56->Location = System::Drawing::Point(252, 236);
			this->label56->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label56->Name = L"label56";
			this->label56->Size = System::Drawing::Size(43, 17);
			this->label56->TabIndex = 187;
			this->label56->Text = L"Luas:";
			// 
			// label57
			// 
			this->label57->AutoSize = true;
			this->label57->Location = System::Drawing::Point(252, 198);
			this->label57->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label57->Name = L"label57";
			this->label57->Size = System::Drawing::Size(30, 17);
			this->label57->TabIndex = 186;
			this->label57->Text = L"Inti:";
			// 
			// textBoxHomogenityPlasma
			// 
			this->textBoxHomogenityPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxHomogenityPlasma->Location = System::Drawing::Point(117, 545);
			this->textBoxHomogenityPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxHomogenityPlasma->Name = L"textBoxHomogenityPlasma";
			this->textBoxHomogenityPlasma->ReadOnly = true;
			this->textBoxHomogenityPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxHomogenityPlasma->TabIndex = 185;
			this->textBoxHomogenityPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label51
			// 
			this->label51->AutoSize = true;
			this->label51->Location = System::Drawing::Point(16, 548);
			this->label51->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label51->Name = L"label51";
			this->label51->Size = System::Drawing::Size(87, 17);
			this->label51->TabIndex = 184;
			this->label51->Text = L"Homogenity:";
			// 
			// textBoxContrastPlasma
			// 
			this->textBoxContrastPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxContrastPlasma->Location = System::Drawing::Point(117, 516);
			this->textBoxContrastPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxContrastPlasma->Name = L"textBoxContrastPlasma";
			this->textBoxContrastPlasma->ReadOnly = true;
			this->textBoxContrastPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxContrastPlasma->TabIndex = 183;
			this->textBoxContrastPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label52
			// 
			this->label52->AutoSize = true;
			this->label52->Location = System::Drawing::Point(16, 520);
			this->label52->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label52->Name = L"label52";
			this->label52->Size = System::Drawing::Size(65, 17);
			this->label52->TabIndex = 182;
			this->label52->Text = L"Contrast:";
			// 
			// label20
			// 
			this->label20->AutoSize = true;
			this->label20->Location = System::Drawing::Point(16, 491);
			this->label20->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label20->Name = L"label20";
			this->label20->Size = System::Drawing::Size(57, 17);
			this->label20->TabIndex = 181;
			this->label20->Text = L"Energy:";
			// 
			// textBoxEnergyPlasma
			// 
			this->textBoxEnergyPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxEnergyPlasma->Location = System::Drawing::Point(117, 488);
			this->textBoxEnergyPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxEnergyPlasma->Name = L"textBoxEnergyPlasma";
			this->textBoxEnergyPlasma->ReadOnly = true;
			this->textBoxEnergyPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxEnergyPlasma->TabIndex = 180;
			this->textBoxEnergyPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxEntropyPlasma
			// 
			this->textBoxEntropyPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxEntropyPlasma->Location = System::Drawing::Point(117, 459);
			this->textBoxEntropyPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxEntropyPlasma->Name = L"textBoxEntropyPlasma";
			this->textBoxEntropyPlasma->ReadOnly = true;
			this->textBoxEntropyPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxEntropyPlasma->TabIndex = 179;
			this->textBoxEntropyPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxBPlasma
			// 
			this->textBoxBPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxBPlasma->Location = System::Drawing::Point(117, 431);
			this->textBoxBPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxBPlasma->Name = L"textBoxBPlasma";
			this->textBoxBPlasma->ReadOnly = true;
			this->textBoxBPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxBPlasma->TabIndex = 178;
			this->textBoxBPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxGPlasma
			// 
			this->textBoxGPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxGPlasma->Location = System::Drawing::Point(117, 403);
			this->textBoxGPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxGPlasma->Name = L"textBoxGPlasma";
			this->textBoxGPlasma->ReadOnly = true;
			this->textBoxGPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxGPlasma->TabIndex = 177;
			this->textBoxGPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxRPlasma
			// 
			this->textBoxRPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxRPlasma->Location = System::Drawing::Point(117, 374);
			this->textBoxRPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxRPlasma->Name = L"textBoxRPlasma";
			this->textBoxRPlasma->ReadOnly = true;
			this->textBoxRPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxRPlasma->TabIndex = 176;
			this->textBoxRPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label27
			// 
			this->label27->AutoSize = true;
			this->label27->Location = System::Drawing::Point(16, 463);
			this->label27->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label27->Name = L"label27";
			this->label27->Size = System::Drawing::Size(61, 17);
			this->label27->TabIndex = 175;
			this->label27->Text = L"Entropy:";
			// 
			// label28
			// 
			this->label28->AutoSize = true;
			this->label28->Location = System::Drawing::Point(16, 435);
			this->label28->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label28->Name = L"label28";
			this->label28->Size = System::Drawing::Size(61, 17);
			this->label28->TabIndex = 174;
			this->label28->Text = L"Kanal B:";
			// 
			// label29
			// 
			this->label29->AutoSize = true;
			this->label29->Location = System::Drawing::Point(16, 406);
			this->label29->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label29->Name = L"label29";
			this->label29->Size = System::Drawing::Size(63, 17);
			this->label29->TabIndex = 173;
			this->label29->Text = L"Kanal G:";
			// 
			// label30
			// 
			this->label30->AutoSize = true;
			this->label30->Location = System::Drawing::Point(16, 378);
			this->label30->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label30->Name = L"label30";
			this->label30->Size = System::Drawing::Size(62, 17);
			this->label30->TabIndex = 172;
			this->label30->Text = L"Kanal R:";
			// 
			// label31
			// 
			this->label31->AutoSize = true;
			this->label31->Location = System::Drawing::Point(16, 349);
			this->label31->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label31->Name = L"label31";
			this->label31->Size = System::Drawing::Size(74, 17);
			this->label31->TabIndex = 171;
			this->label31->Text = L"Circularity:";
			// 
			// textBoxCircularityPlasma
			// 
			this->textBoxCircularityPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxCircularityPlasma->Location = System::Drawing::Point(118, 345);
			this->textBoxCircularityPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxCircularityPlasma->Name = L"textBoxCircularityPlasma";
			this->textBoxCircularityPlasma->ReadOnly = true;
			this->textBoxCircularityPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxCircularityPlasma->TabIndex = 170;
			this->textBoxCircularityPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxGranularityPlasma
			// 
			this->textBoxGranularityPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxGranularityPlasma->Location = System::Drawing::Point(118, 317);
			this->textBoxGranularityPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxGranularityPlasma->Name = L"textBoxGranularityPlasma";
			this->textBoxGranularityPlasma->ReadOnly = true;
			this->textBoxGranularityPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxGranularityPlasma->TabIndex = 169;
			this->textBoxGranularityPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxSolidityPlasma
			// 
			this->textBoxSolidityPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxSolidityPlasma->Location = System::Drawing::Point(118, 289);
			this->textBoxSolidityPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxSolidityPlasma->Name = L"textBoxSolidityPlasma";
			this->textBoxSolidityPlasma->ReadOnly = true;
			this->textBoxSolidityPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxSolidityPlasma->TabIndex = 168;
			this->textBoxSolidityPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxKelilingPlasma
			// 
			this->textBoxKelilingPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxKelilingPlasma->Location = System::Drawing::Point(118, 260);
			this->textBoxKelilingPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxKelilingPlasma->Name = L"textBoxKelilingPlasma";
			this->textBoxKelilingPlasma->ReadOnly = true;
			this->textBoxKelilingPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxKelilingPlasma->TabIndex = 167;
			this->textBoxKelilingPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxLuasPlasma
			// 
			this->textBoxLuasPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxLuasPlasma->Location = System::Drawing::Point(118, 232);
			this->textBoxLuasPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxLuasPlasma->Name = L"textBoxLuasPlasma";
			this->textBoxLuasPlasma->ReadOnly = true;
			this->textBoxLuasPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxLuasPlasma->TabIndex = 166;
			this->textBoxLuasPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label32
			// 
			this->label32->AutoSize = true;
			this->label32->Location = System::Drawing::Point(16, 321);
			this->label32->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label32->Name = L"label32";
			this->label32->Size = System::Drawing::Size(82, 17);
			this->label32->TabIndex = 165;
			this->label32->Text = L"Granularity:";
			// 
			// label33
			// 
			this->label33->AutoSize = true;
			this->label33->Location = System::Drawing::Point(16, 292);
			this->label33->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label33->Name = L"label33";
			this->label33->Size = System::Drawing::Size(57, 17);
			this->label33->TabIndex = 164;
			this->label33->Text = L"Solidity:";
			// 
			// label34
			// 
			this->label34->AutoSize = true;
			this->label34->Location = System::Drawing::Point(16, 264);
			this->label34->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label34->Name = L"label34";
			this->label34->Size = System::Drawing::Size(57, 17);
			this->label34->TabIndex = 163;
			this->label34->Text = L"Keliling:";
			// 
			// label41
			// 
			this->label41->AutoSize = true;
			this->label41->Location = System::Drawing::Point(16, 236);
			this->label41->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label41->Name = L"label41";
			this->label41->Size = System::Drawing::Size(43, 17);
			this->label41->TabIndex = 162;
			this->label41->Text = L"Luas:";
			// 
			// label49
			// 
			this->label49->AutoSize = true;
			this->label49->Location = System::Drawing::Point(16, 198);
			this->label49->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label49->Name = L"label49";
			this->label49->Size = System::Drawing::Size(81, 17);
			this->label49->TabIndex = 161;
			this->label49->Text = L"Sitoplasma:";
			// 
			// textBoxLuasNormalisasiInti
			// 
			this->textBoxLuasNormalisasiInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxLuasNormalisasiInti->Location = System::Drawing::Point(118, 34);
			this->textBoxLuasNormalisasiInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxLuasNormalisasiInti->Name = L"textBoxLuasNormalisasiInti";
			this->textBoxLuasNormalisasiInti->ReadOnly = true;
			this->textBoxLuasNormalisasiInti->Size = System::Drawing::Size(95, 22);
			this->textBoxLuasNormalisasiInti->TabIndex = 160;
			this->textBoxLuasNormalisasiInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label38
			// 
			this->label38->AutoSize = true;
			this->label38->Location = System::Drawing::Point(16, 39);
			this->label38->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label38->Name = L"label38";
			this->label38->Size = System::Drawing::Size(81, 17);
			this->label38->TabIndex = 159;
			this->label38->Text = L"Luas Norm:";
			// 
			// textBoxEccentricity
			// 
			this->textBoxEccentricity->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxEccentricity->Location = System::Drawing::Point(118, 94);
			this->textBoxEccentricity->Margin = System::Windows::Forms::Padding(4);
			this->textBoxEccentricity->Name = L"textBoxEccentricity";
			this->textBoxEccentricity->ReadOnly = true;
			this->textBoxEccentricity->Size = System::Drawing::Size(95, 22);
			this->textBoxEccentricity->TabIndex = 158;
			this->textBoxEccentricity->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxKelilingNormalisasiInti
			// 
			this->textBoxKelilingNormalisasiInti->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxKelilingNormalisasiInti->Location = System::Drawing::Point(118, 66);
			this->textBoxKelilingNormalisasiInti->Margin = System::Windows::Forms::Padding(4);
			this->textBoxKelilingNormalisasiInti->Name = L"textBoxKelilingNormalisasiInti";
			this->textBoxKelilingNormalisasiInti->ReadOnly = true;
			this->textBoxKelilingNormalisasiInti->Size = System::Drawing::Size(95, 22);
			this->textBoxKelilingNormalisasiInti->TabIndex = 157;
			this->textBoxKelilingNormalisasiInti->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label35
			// 
			this->label35->AutoSize = true;
			this->label35->Location = System::Drawing::Point(16, 100);
			this->label35->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label35->Name = L"label35";
			this->label35->Size = System::Drawing::Size(84, 17);
			this->label35->TabIndex = 156;
			this->label35->Text = L"Eccentricity:";
			// 
			// label37
			// 
			this->label37->AutoSize = true;
			this->label37->Location = System::Drawing::Point(16, 71);
			this->label37->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label37->Name = L"label37";
			this->label37->Size = System::Drawing::Size(95, 17);
			this->label37->TabIndex = 155;
			this->label37->Text = L"Keliling Norm:";
			// 
			// textBoxKelilingIntiPlasma
			// 
			this->textBoxKelilingIntiPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxKelilingIntiPlasma->Location = System::Drawing::Point(118, 152);
			this->textBoxKelilingIntiPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxKelilingIntiPlasma->Name = L"textBoxKelilingIntiPlasma";
			this->textBoxKelilingIntiPlasma->ReadOnly = true;
			this->textBoxKelilingIntiPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxKelilingIntiPlasma->TabIndex = 154;
			this->textBoxKelilingIntiPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// textBoxLuasIntiPlasma
			// 
			this->textBoxLuasIntiPlasma->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxLuasIntiPlasma->Location = System::Drawing::Point(118, 123);
			this->textBoxLuasIntiPlasma->Margin = System::Windows::Forms::Padding(4);
			this->textBoxLuasIntiPlasma->Name = L"textBoxLuasIntiPlasma";
			this->textBoxLuasIntiPlasma->ReadOnly = true;
			this->textBoxLuasIntiPlasma->Size = System::Drawing::Size(95, 22);
			this->textBoxLuasIntiPlasma->TabIndex = 153;
			this->textBoxLuasIntiPlasma->TextAlign = System::Windows::Forms::HorizontalAlignment::Right;
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(16, 157);
			this->label4->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(46, 17);
			this->label4->TabIndex = 152;
			this->label4->Text = L"KI/KP:";
			// 
			// label26
			// 
			this->label26->AutoSize = true;
			this->label26->Location = System::Drawing::Point(16, 129);
			this->label26->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label26->Name = L"label26";
			this->label26->Size = System::Drawing::Size(44, 17);
			this->label26->TabIndex = 151;
			this->label26->Text = L"LI/LP:";
			// 
			// groupBox3
			// 
			this->groupBox3->Controls->Add(this->pictureBoxRGB);
			this->groupBox3->Controls->Add(this->pictureBoxSitoplasma);
			this->groupBox3->Controls->Add(this->label48);
			this->groupBox3->Controls->Add(this->pictureBoxInti);
			this->groupBox3->Controls->Add(this->label47);
			this->groupBox3->Controls->Add(this->label46);
			this->groupBox3->Location = System::Drawing::Point(15, 20);
			this->groupBox3->Name = L"groupBox3";
			this->groupBox3->Size = System::Drawing::Size(431, 194);
			this->groupBox3->TabIndex = 153;
			this->groupBox3->TabStop = false;
			this->groupBox3->Text = L"Citra Sel Darah Putih:";
			// 
			// pictureBoxRGB
			// 
			this->pictureBoxRGB->BackColor = System::Drawing::SystemColors::Window;
			this->pictureBoxRGB->Location = System::Drawing::Point(7, 46);
			this->pictureBoxRGB->Margin = System::Windows::Forms::Padding(4);
			this->pictureBoxRGB->Name = L"pictureBoxRGB";
			this->pictureBoxRGB->Size = System::Drawing::Size(130, 130);
			this->pictureBoxRGB->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBoxRGB->TabIndex = 101;
			this->pictureBoxRGB->TabStop = false;
			// 
			// pictureBoxSitoplasma
			// 
			this->pictureBoxSitoplasma->BackColor = System::Drawing::SystemColors::Window;
			this->pictureBoxSitoplasma->Location = System::Drawing::Point(145, 47);
			this->pictureBoxSitoplasma->Margin = System::Windows::Forms::Padding(4);
			this->pictureBoxSitoplasma->Name = L"pictureBoxSitoplasma";
			this->pictureBoxSitoplasma->Size = System::Drawing::Size(130, 130);
			this->pictureBoxSitoplasma->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBoxSitoplasma->TabIndex = 100;
			this->pictureBoxSitoplasma->TabStop = false;
			// 
			// label48
			// 
			this->label48->AutoSize = true;
			this->label48->Location = System::Drawing::Point(11, 26);
			this->label48->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label48->Name = L"label48";
			this->label48->Size = System::Drawing::Size(42, 17);
			this->label48->TabIndex = 102;
			this->label48->Text = L"RGB:";
			// 
			// pictureBoxInti
			// 
			this->pictureBoxInti->BackColor = System::Drawing::SystemColors::Window;
			this->pictureBoxInti->Location = System::Drawing::Point(283, 47);
			this->pictureBoxInti->Margin = System::Windows::Forms::Padding(4);
			this->pictureBoxInti->Name = L"pictureBoxInti";
			this->pictureBoxInti->Size = System::Drawing::Size(130, 130);
			this->pictureBoxInti->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBoxInti->TabIndex = 103;
			this->pictureBoxInti->TabStop = false;
			// 
			// label47
			// 
			this->label47->AutoSize = true;
			this->label47->Location = System::Drawing::Point(142, 26);
			this->label47->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label47->Name = L"label47";
			this->label47->Size = System::Drawing::Size(81, 17);
			this->label47->TabIndex = 104;
			this->label47->Text = L"Sitoplasma:";
			// 
			// label46
			// 
			this->label46->AutoSize = true;
			this->label46->Location = System::Drawing::Point(280, 26);
			this->label46->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label46->Name = L"label46";
			this->label46->Size = System::Drawing::Size(30, 17);
			this->label46->TabIndex = 105;
			this->label46->Text = L"Inti:";
			// 
			// groupBox2
			// 
			this->groupBox2->Controls->Add(this->label5);
			this->groupBox2->Controls->Add(this->textBoxIntiS);
			this->groupBox2->Controls->Add(this->hScrollBarIntiS);
			this->groupBox2->Controls->Add(this->label9);
			this->groupBox2->Controls->Add(this->label21);
			this->groupBox2->Controls->Add(this->textBoxPlasmaV);
			this->groupBox2->Controls->Add(this->hScrollBarPlasmaV);
			this->groupBox2->Controls->Add(this->label23);
			this->groupBox2->Controls->Add(this->textBoxPlasmaS);
			this->groupBox2->Controls->Add(this->hScrollBarPlasmaS);
			this->groupBox2->Controls->Add(this->label24);
			this->groupBox2->Controls->Add(this->textBoxPlasmaH);
			this->groupBox2->Controls->Add(this->hScrollBarPlasmaH);
			this->groupBox2->Controls->Add(this->label25);
			this->groupBox2->Location = System::Drawing::Point(22, 356);
			this->groupBox2->Name = L"groupBox2";
			this->groupBox2->Size = System::Drawing::Size(393, 201);
			this->groupBox2->TabIndex = 152;
			this->groupBox2->TabStop = false;
			this->groupBox2->Text = L"Pengaturan Pengambangan:";
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(7, 140);
			this->label5->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(30, 17);
			this->label5->TabIndex = 153;
			this->label5->Text = L"Inti:";
			// 
			// textBoxIntiS
			// 
			this->textBoxIntiS->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxIntiS->Location = System::Drawing::Point(39, 163);
			this->textBoxIntiS->Margin = System::Windows::Forms::Padding(4);
			this->textBoxIntiS->Name = L"textBoxIntiS";
			this->textBoxIntiS->ReadOnly = true;
			this->textBoxIntiS->Size = System::Drawing::Size(59, 22);
			this->textBoxIntiS->TabIndex = 152;
			this->textBoxIntiS->Text = L"149";
			this->textBoxIntiS->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// hScrollBarIntiS
			// 
			this->hScrollBarIntiS->Location = System::Drawing::Point(103, 164);
			this->hScrollBarIntiS->Maximum = 264;
			this->hScrollBarIntiS->Name = L"hScrollBarIntiS";
			this->hScrollBarIntiS->Size = System::Drawing::Size(275, 17);
			this->hScrollBarIntiS->TabIndex = 151;
			this->hScrollBarIntiS->Value = 149;
			this->hScrollBarIntiS->Scroll += gcnew System::Windows::Forms::ScrollEventHandler(this, &MyForm::hScrollBarIntiS_Scroll);
			// 
			// label9
			// 
			this->label9->AutoSize = true;
			this->label9->Location = System::Drawing::Point(7, 166);
			this->label9->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label9->Name = L"label9";
			this->label9->Size = System::Drawing::Size(21, 17);
			this->label9->TabIndex = 150;
			this->label9->Text = L"S:";
			// 
			// label21
			// 
			this->label21->AutoSize = true;
			this->label21->Location = System::Drawing::Point(7, 30);
			this->label21->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label21->Name = L"label21";
			this->label21->Size = System::Drawing::Size(81, 17);
			this->label21->TabIndex = 149;
			this->label21->Text = L"Sitoplasma:";
			// 
			// textBoxPlasmaV
			// 
			this->textBoxPlasmaV->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxPlasmaV->Location = System::Drawing::Point(39, 108);
			this->textBoxPlasmaV->Margin = System::Windows::Forms::Padding(4);
			this->textBoxPlasmaV->Name = L"textBoxPlasmaV";
			this->textBoxPlasmaV->ReadOnly = true;
			this->textBoxPlasmaV->Size = System::Drawing::Size(59, 22);
			this->textBoxPlasmaV->TabIndex = 148;
			this->textBoxPlasmaV->Text = L"62";
			this->textBoxPlasmaV->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// hScrollBarPlasmaV
			// 
			this->hScrollBarPlasmaV->Location = System::Drawing::Point(103, 110);
			this->hScrollBarPlasmaV->Maximum = 264;
			this->hScrollBarPlasmaV->Name = L"hScrollBarPlasmaV";
			this->hScrollBarPlasmaV->Size = System::Drawing::Size(275, 17);
			this->hScrollBarPlasmaV->TabIndex = 147;
			this->hScrollBarPlasmaV->Value = 62;
			this->hScrollBarPlasmaV->Scroll += gcnew System::Windows::Forms::ScrollEventHandler(this, &MyForm::hScrollBarPlasmaV_Scroll);
			// 
			// label23
			// 
			this->label23->AutoSize = true;
			this->label23->Location = System::Drawing::Point(7, 112);
			this->label23->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label23->Name = L"label23";
			this->label23->Size = System::Drawing::Size(21, 17);
			this->label23->TabIndex = 146;
			this->label23->Text = L"V:";
			// 
			// textBoxPlasmaS
			// 
			this->textBoxPlasmaS->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxPlasmaS->Location = System::Drawing::Point(39, 80);
			this->textBoxPlasmaS->Margin = System::Windows::Forms::Padding(4);
			this->textBoxPlasmaS->Name = L"textBoxPlasmaS";
			this->textBoxPlasmaS->ReadOnly = true;
			this->textBoxPlasmaS->Size = System::Drawing::Size(59, 22);
			this->textBoxPlasmaS->TabIndex = 145;
			this->textBoxPlasmaS->Text = L"27";
			this->textBoxPlasmaS->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// hScrollBarPlasmaS
			// 
			this->hScrollBarPlasmaS->Location = System::Drawing::Point(103, 81);
			this->hScrollBarPlasmaS->Maximum = 264;
			this->hScrollBarPlasmaS->Name = L"hScrollBarPlasmaS";
			this->hScrollBarPlasmaS->Size = System::Drawing::Size(275, 17);
			this->hScrollBarPlasmaS->TabIndex = 144;
			this->hScrollBarPlasmaS->Value = 27;
			this->hScrollBarPlasmaS->Scroll += gcnew System::Windows::Forms::ScrollEventHandler(this, &MyForm::hScrollBarPlasmaS_Scroll);
			// 
			// label24
			// 
			this->label24->AutoSize = true;
			this->label24->Location = System::Drawing::Point(7, 84);
			this->label24->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label24->Name = L"label24";
			this->label24->Size = System::Drawing::Size(21, 17);
			this->label24->TabIndex = 143;
			this->label24->Text = L"S:";
			// 
			// textBoxPlasmaH
			// 
			this->textBoxPlasmaH->BackColor = System::Drawing::SystemColors::Window;
			this->textBoxPlasmaH->Location = System::Drawing::Point(39, 52);
			this->textBoxPlasmaH->Margin = System::Windows::Forms::Padding(4);
			this->textBoxPlasmaH->Name = L"textBoxPlasmaH";
			this->textBoxPlasmaH->ReadOnly = true;
			this->textBoxPlasmaH->Size = System::Drawing::Size(59, 22);
			this->textBoxPlasmaH->TabIndex = 142;
			this->textBoxPlasmaH->Text = L"130";
			this->textBoxPlasmaH->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// hScrollBarPlasmaH
			// 
			this->hScrollBarPlasmaH->Location = System::Drawing::Point(103, 53);
			this->hScrollBarPlasmaH->Maximum = 189;
			this->hScrollBarPlasmaH->Name = L"hScrollBarPlasmaH";
			this->hScrollBarPlasmaH->Size = System::Drawing::Size(275, 17);
			this->hScrollBarPlasmaH->TabIndex = 141;
			this->hScrollBarPlasmaH->Value = 130;
			this->hScrollBarPlasmaH->Scroll += gcnew System::Windows::Forms::ScrollEventHandler(this, &MyForm::hScrollBarPlasmaH_Scroll);
			// 
			// label25
			// 
			this->label25->AutoSize = true;
			this->label25->Location = System::Drawing::Point(7, 56);
			this->label25->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label25->Name = L"label25";
			this->label25->Size = System::Drawing::Size(22, 17);
			this->label25->TabIndex = 140;
			this->label25->Text = L"H:";
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->radioButtonConvexArea);
			this->groupBox1->Controls->Add(this->radioButtonConvexHull);
			this->groupBox1->Controls->Add(this->radioButtonKontur);
			this->groupBox1->Controls->Add(this->radioButtonBiner);
			this->groupBox1->Location = System::Drawing::Point(21, 239);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(217, 92);
			this->groupBox1->TabIndex = 151;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Opsi Tampilan:";
			// 
			// radioButtonConvexArea
			// 
			this->radioButtonConvexArea->AutoSize = true;
			this->radioButtonConvexArea->Location = System::Drawing::Point(95, 60);
			this->radioButtonConvexArea->Margin = System::Windows::Forms::Padding(4);
			this->radioButtonConvexArea->Name = L"radioButtonConvexArea";
			this->radioButtonConvexArea->Size = System::Drawing::Size(109, 21);
			this->radioButtonConvexArea->TabIndex = 114;
			this->radioButtonConvexArea->Text = L"Convex Area";
			this->radioButtonConvexArea->UseVisualStyleBackColor = true;
			this->radioButtonConvexArea->CheckedChanged += gcnew System::EventHandler(this, &MyForm::radioButtonConvexArea_CheckedChanged);
			// 
			// radioButtonConvexHull
			// 
			this->radioButtonConvexHull->AutoSize = true;
			this->radioButtonConvexHull->Location = System::Drawing::Point(95, 31);
			this->radioButtonConvexHull->Margin = System::Windows::Forms::Padding(4);
			this->radioButtonConvexHull->Name = L"radioButtonConvexHull";
			this->radioButtonConvexHull->Size = System::Drawing::Size(103, 21);
			this->radioButtonConvexHull->TabIndex = 113;
			this->radioButtonConvexHull->Text = L"Convex Hull";
			this->radioButtonConvexHull->UseVisualStyleBackColor = true;
			this->radioButtonConvexHull->CheckedChanged += gcnew System::EventHandler(this, &MyForm::radioButtonConvexHull_CheckedChanged);
			// 
			// radioButtonKontur
			// 
			this->radioButtonKontur->AutoSize = true;
			this->radioButtonKontur->Location = System::Drawing::Point(13, 59);
			this->radioButtonKontur->Margin = System::Windows::Forms::Padding(4);
			this->radioButtonKontur->Name = L"radioButtonKontur";
			this->radioButtonKontur->Size = System::Drawing::Size(71, 21);
			this->radioButtonKontur->TabIndex = 112;
			this->radioButtonKontur->Text = L"Kontur";
			this->radioButtonKontur->UseVisualStyleBackColor = true;
			this->radioButtonKontur->CheckedChanged += gcnew System::EventHandler(this, &MyForm::radioButtonKontur_CheckedChanged);
			// 
			// radioButtonBiner
			// 
			this->radioButtonBiner->AutoSize = true;
			this->radioButtonBiner->Checked = true;
			this->radioButtonBiner->Location = System::Drawing::Point(13, 31);
			this->radioButtonBiner->Margin = System::Windows::Forms::Padding(4);
			this->radioButtonBiner->Name = L"radioButtonBiner";
			this->radioButtonBiner->Size = System::Drawing::Size(62, 21);
			this->radioButtonBiner->TabIndex = 111;
			this->radioButtonBiner->TabStop = true;
			this->radioButtonBiner->Text = L"Biner";
			this->radioButtonBiner->UseVisualStyleBackColor = true;
			this->radioButtonBiner->CheckedChanged += gcnew System::EventHandler(this, &MyForm::radioButtonBiner_CheckedChanged);
			// 
			// openFileDialogCitra
			// 
			this->openFileDialogCitra->FileName = L"openFileDialog1";
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1105, 746);
			this->Controls->Add(this->tabControl1);
			this->Controls->Add(this->statusStrip1);
			this->Controls->Add(this->menuStrip1);
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->Margin = System::Windows::Forms::Padding(4);
			this->Name = L"MyForm";
			this->Text = L"Citra Darah";
			this->Load += gcnew System::EventHandler(this, &MyForm::MyForm_Load);
			this->statusStrip1->ResumeLayout(false);
			this->statusStrip1->PerformLayout();
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			this->tabControl1->ResumeLayout(false);
			this->tabPageLatih->ResumeLayout(false);
			this->tabPageLatih->PerformLayout();
			this->groupBoxMLP->ResumeLayout(false);
			this->groupBoxMLP->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownBykHdn))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownHdn5))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownHdn4))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownHdn3))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownHdn2))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownOutput))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownHdn1))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDownInputLayer))->EndInit();
			this->groupBoxSVM->ResumeLayout(false);
			this->groupBoxSVM->PerformLayout();
			this->groupBoxSupervisedData->ResumeLayout(false);
			this->groupBoxSupervisedData->PerformLayout();
			this->tabPageEkstraksiFitur->ResumeLayout(false);
			this->groupBox4->ResumeLayout(false);
			this->groupBox4->PerformLayout();
			this->groupBox3->ResumeLayout(false);
			this->groupBox3->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxRGB))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxSitoplasma))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxInti))->EndInit();
			this->groupBox2->ResumeLayout(false);
			this->groupBox2->PerformLayout();
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
		//Fungsi untuk mengonversi format string dari pustaka C++ ke pustaka OpenCV
		std::string MarshalString(System::String ^ s) {
			using namespace System::Runtime::InteropServices;
			const char* chars = (const char*)(Marshal::StringToHGlobalAnsi(s)).ToPointer();
			string os = chars;
			Marshal::FreeHGlobal(System::IntPtr((void*)chars));
			return os;
		}

		//Fungsi Pengolahan Citra untuk Ekstraksi
		void ProsesCitra(int mode) {
			if (citraAwal.data == 0)
				return;
			//Proses konversi RGB ke HSV
			Mat citraHSV, citraSplitHSV[3], citraSplitRGB[3], citraH, citraS, citraV, citraR, citraG, citraB, citraGray;
			cvtColor(citraAwal, citraHSV, CV_BGR2HSV); //Konversi RGB ke HSV
			cvtColor(citraAwal, citraGray, CV_BGR2GRAY); //Konversi RGB ke GRAYSCALE
			split(citraHSV, citraSplitHSV); //Pemisahan kanal HSV
			split(citraAwal, citraSplitRGB); //Pemisahan kanal RGB
			citraH = citraSplitHSV[0]; //Akses kanal H dari HSV
			citraS = citraSplitHSV[1]; //Akses kanal S dari HSV
			citraV = citraSplitHSV[2]; //Akses kanal V dari HSV

			citraR = citraSplitRGB[0]; //Akses kanal R dari RGB
			citraG = citraSplitRGB[1]; //Akses kanal G dari RGB
			citraB = citraSplitRGB[2]; //Akses kanal B dari RGB

									   //Proses iteratif nilai ambang optimum
			Mat ambangTemp;
			float tPlasmaH0 = tPlasmaH, tPlasmaS0 = tPlasmaS, tPlasmaV0 = tPlasmaV;
			for (int i = 0; i < 1; i++) { //iteratif nilai ambang sitoplasma kanal H
				threshold(citraH, ambangTemp, tPlasmaH0, 255, THRESH_BINARY);
				int g1 = 0, g2 = 0;
				float mu1 = 0, mu2 = 0, tPlasmaH1;
				for (int x = 0; x < citraH.rows; x++) {
					for (int y = 0; y < citraH.cols; y++) {
						if (citraH.at<uchar>(x, y) > tPlasmaH0) {
							mu1 += citraH.at<uchar>(x, y);
							g1++;
						}
						else if (citraH.at<uchar>(x, y) <= tPlasmaH0) {
							mu2 += citraH.at<uchar>(x, y);
							g2++;
						}
					}
				}
				mu1 = mu1 / g1;
				mu2 = mu2 / g2;
				tPlasmaH1 = (mu1 + mu2) / 2;
				if (tPlasmaH0 > tPlasmaH1) {
					i--;
					tPlasmaH0 = tPlasmaH1;
				}
				tPlasmaH1 = floor(tPlasmaH1);
				temp_tPlasmaH = tPlasmaH1;
			}
			for (int i = 0; i < 1; i++) { //iteratif nilai ambang sitoplasma kanal S
				threshold(citraS, ambangTemp, tPlasmaS0, 255, THRESH_BINARY);
				int g1 = 0, g2 = 0;
				float mu1 = 0, mu2 = 0, tPlasmaS1;
				for (int x = 0; x < citraS.rows; x++) {
					for (int y = 0; y < citraS.cols; y++) {
						if (citraS.at<uchar>(x, y) > tPlasmaS0) {
							mu1 += citraS.at<uchar>(x, y);
							g1++;
						}
						else if (citraS.at<uchar>(x, y) <= tPlasmaS0) {
							mu2 += citraS.at<uchar>(x, y);
							g2++;
						}
					}
				}
				mu1 = mu1 / g1;
				mu2 = mu2 / g2;
				tPlasmaS1 = (mu1 + mu2) / 2;
				if (tPlasmaS0 > tPlasmaS1) {
					i--;
					tPlasmaS0 = tPlasmaS1;
				}
				tPlasmaS1 = floor(tPlasmaS1);
				temp_tPlasmaS = tPlasmaS1;
			}
			for (int i = 0; i < 1; i++) { //iteratif nilai ambang sitoplasma kanal V
				threshold(citraV, ambangTemp, tPlasmaV0, 255, THRESH_BINARY);
				int g1 = 0, g2 = 0;
				float mu1 = 0, mu2 = 0, tPlasmaV1;
				for (int x = 0; x < citraV.rows; x++) {
					for (int y = 0; y < citraV.cols; y++) {
						if (citraV.at<uchar>(x, y) > tPlasmaV0) {
							mu1 += citraV.at<uchar>(x, y);
							g1++;
						}
						else if (citraV.at<uchar>(x, y) <= tPlasmaV0) {
							mu2 += citraV.at<uchar>(x, y);
							g2++;
						}
					}
				}
				mu1 = mu1 / g1;
				mu2 = mu2 / g2;
				if (g2 == 0) {
					tPlasmaV1 = mu1 / 2;
				}
				else {
					tPlasmaV1 = (mu1 + mu2) / 2;
				}
				if (tPlasmaV0 > tPlasmaV1) {
					i--;
					tPlasmaV0 = tPlasmaV1;
				}
				tPlasmaV1 = floor(tPlasmaV1);
				temp_tPlasmaV = tPlasmaV1;
			}
			float tIntiS0 = tIntiS;
			for (int i = 0; i < 1; i++) { //iteratif nilai ambang inti kanal S
				threshold(citraS, ambangTemp, tIntiS0, 255, THRESH_BINARY);
				int g1 = 0, g2 = 0;
				float mu1 = 0, mu2 = 0, tIntiS1;
				for (int x = 0; x < citraS.rows; x++) {
					for (int y = 0; y < citraS.cols; y++) {
						if (citraS.at<uchar>(x, y) > tIntiS0) {
							mu1 += citraS.at<uchar>(x, y);
							g1++;
						}
						else if (citraS.at<uchar>(x, y) <= tIntiS0) {
							mu2 += citraS.at<uchar>(x, y);
							g2++;
						}
					}
				}
				mu1 = mu1 / g1;
				mu2 = mu2 / g2;
				tIntiS1 = (mu1 + mu2) / 2;
				if (tIntiS0 > tIntiS1) {
					i--;
					tIntiS0 = tIntiS1;
				}
				tIntiS1 = floor(tIntiS1);
				temp_tIntiS = tIntiS1;
			}
			//Pengambangan sitoplasma dan inti
			Mat ambangPlasmaH, ambangPlasmaS, ambangPlasmaV, ambangIntiS, ambangPlasmaHasil, ambangIntiHasil;
			threshold(citraH, ambangPlasmaH, tPlasmaH, 255, THRESH_BINARY);
			threshold(citraS, ambangPlasmaS, tPlasmaS, 255, THRESH_BINARY);
			threshold(citraV, ambangPlasmaV, tPlasmaV, 255, THRESH_BINARY);
			threshold(citraS, ambangIntiS, tIntiS, 255, THRESH_BINARY);
			bitwise_and(ambangPlasmaH, ambangPlasmaS, ambangPlasmaHasil);
			bitwise_and(ambangPlasmaV, ambangPlasmaHasil, ambangPlasmaHasil);
			ambangIntiHasil = ambangIntiS.clone();

			//Menghitung luas sitoplasma
			luasPlasma = countNonZero(ambangPlasmaHasil);
			//Menghitung luas inti
			luasInti = countNonZero(ambangIntiHasil);

			textBoxLuasPlasma->Text = luasPlasma.ToString();
			textBoxLuasInti->Text = luasInti.ToString();
			//Deteksi garis kontur tepi sitoplasma dan inti
			vector<vector<cv::Point>> konturPlasma, konturInti;
			vector<Vec4i> hierarkiKonturPlasma, hierarkiKonturInti;
			Mat konturTemp = ambangPlasmaHasil.clone();
			findContours(konturTemp, konturPlasma, hierarkiKonturPlasma, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
			konturTemp = ambangIntiHasil.clone();
			findContours(konturTemp, konturInti, hierarkiKonturInti, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
			//Pembentukan convex hull dari garis kontur tepi sitoplasma dan inti
			//Beserta pengisian convex hull sehingga membentuk area hull
			Mat citraKonturPlasma = Mat::zeros(ambangPlasmaHasil.rows, ambangPlasmaHasil.cols, CV_8UC3);
			Mat citraKonturInti = Mat::zeros(ambangIntiHasil.rows, ambangIntiHasil.cols, CV_8UC3);
			Mat citraHullPlasma = citraKonturPlasma.clone();
			Mat citraHullInti = citraKonturInti.clone();
			Mat citraAreaHullPlasma = citraKonturPlasma.clone();
			Mat citraAreaHullInti = citraKonturInti.clone();
			vector<vector<cv::Point>> hullPlasma(konturPlasma.size());
			vector<vector<cv::Point>> hullInti(konturInti.size());
			int luasConvexPlasma, luasConvexInti;
			for (int i = 0; i < konturPlasma.size(); i++) {
				convexHull(Mat(konturPlasma[i]), hullPlasma[i], false);
			}
			for (int i = 0; i < konturInti.size(); i++) {
				convexHull(Mat(konturInti[i]), hullInti[i], false);
			}
			for (int i = 0; i < konturPlasma.size(); i++) {
				drawContours(citraKonturPlasma, konturPlasma, i, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, cv::Point());
				drawContours(citraHullPlasma, hullPlasma, i, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, cv::Point());
				cv::Point titikHullPlasma[200];
				int j;
				for (j = 0; j < hullPlasma[i].size(); j++) {
					titikHullPlasma[j] = hullPlasma[i][j];
				}
				const cv::Point* jumlahHullPlasma[1] = { titikHullPlasma };
				int jumlahTitikHullPlasma[] = { j };
				fillPoly(citraAreaHullPlasma, jumlahHullPlasma, jumlahTitikHullPlasma, 1, Scalar(255, 255, 255), 8);
			}
			for (int i = 0; i < konturInti.size(); i++) {
				drawContours(citraKonturInti, konturInti, i, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, cv::Point());
				drawContours(citraHullInti, hullInti, i, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, cv::Point());
				cv::Point titikHullInti[200];
				int j;
				for (j = 0; j < hullInti[i].size(); j++) {
					titikHullInti[j] = hullInti[i][j];
				}
				const cv::Point* jumlahHullInti[1] = { titikHullInti };
				int jumlahTitikHullInti[] = { j };
				fillPoly(citraAreaHullInti, jumlahHullInti, jumlahTitikHullInti, 1, Scalar(255, 255, 255), 8);
			}
			kelilingPlasma = Keliling(citraKonturPlasma);
			kelilingInti = Keliling(citraKonturInti);
			luasConvexPlasma = LuasConvex(citraAreaHullPlasma);
			luasConvexInti = LuasConvex(citraAreaHullInti);
			solidityPlasma = (Convert::ToDouble(luasPlasma)) / (Convert::ToDouble(luasConvexPlasma));
			solidityInti = (Convert::ToDouble(luasInti)) / (Convert::ToDouble(luasConvexInti));

			textBoxKelilingPlasma->Text = kelilingPlasma.ToString();
			textBoxKelilingInti->Text = kelilingInti.ToString();
			textBoxSolidityPlasma->Text = solidityPlasma.ToString();
			textBoxSolidityInti->Text = solidityInti.ToString();

			//Modifikasi citra sitoplasma dan inti
			Mat stddevPlasma, stddevInti, mean, citraModifikasiPlasma, citraModifikasiPlasmaFix, citraModifikasiInti, citraModifikasiIntiFix, citraModPlasmaR, citraModPlasmaG, citraModPlasmaB, citraModIntiR, citraModIntiG, citraModIntiB, citraModPlasmaGray, citraModIntiGray;
			float rerataInti, rerataPlasma, rerataIntiR, rerataIntiG, rerataIntiB, rerataPlasmaR, rerataPlasmaG, rerataPlasmaB, nilaiSDInti, nilaiSDPlasma;
			Scalar rataInti, sdInti, rataIntiR, sdIntiR, rataIntiG, sdIntiG, rataIntiB, sdIntiB; //Untuk Inti
			Scalar rataPlasma, sdPlasma, rataPlasmaR, sdPlasmaR, rataPlasmaG, sdPlasmaG, rataPlasmaB, sdPlasmaB; // Untuk Plasma

																												 //Mengambil citra plasma
			bitwise_not(ambangIntiHasil, citraModifikasiPlasma);
			bitwise_and(citraModifikasiPlasma, ambangPlasmaHasil, citraModifikasiPlasma);
			bitwise_and(citraModifikasiPlasma, citraV, citraModifikasiPlasmaFix); //Kunci di sini

																				  //Untuk Citra Plasma Gray
			bitwise_and(citraModifikasiPlasma, citraGray, citraModPlasmaGray);

			//Untuk Citra Plasma RGB
			bitwise_and(citraModifikasiPlasma, citraR, citraModPlasmaR);
			bitwise_and(citraModifikasiPlasma, citraG, citraModPlasmaG);
			bitwise_and(citraModifikasiPlasma, citraB, citraModPlasmaB);

			//Mengambil citra inti
			bitwise_and(ambangIntiHasil, citraV, citraModifikasiIntiFix);

			//Untuk Citra Inti Gray
			bitwise_and(ambangIntiHasil, citraGray, citraModIntiGray);

			//Untuk Citra Inti RGB
			bitwise_and(ambangIntiHasil, citraR, citraModIntiR);
			bitwise_and(ambangIntiHasil, citraG, citraModIntiG);
			bitwise_and(ambangIntiHasil, citraB, citraModIntiB);

			//rerata = Rerata(citraModifikasiPlasma);
			//nilaiStddevPlasma = StdDev(citraModifikasiPlasma, rerata);
			meanStdDev(citraModifikasiPlasmaFix, rataPlasma, sdPlasma);
			meanStdDev(citraModPlasmaR, rataPlasmaR, sdPlasmaR);
			meanStdDev(citraModPlasmaG, rataPlasmaG, sdPlasmaG);
			meanStdDev(citraModPlasmaB, rataPlasmaB, sdPlasmaB);

			meanStdDev(citraModifikasiIntiFix, rataInti, sdInti);
			meanStdDev(citraModIntiR, rataIntiR, sdIntiR);
			meanStdDev(citraModIntiG, rataIntiG, sdIntiG);
			meanStdDev(citraModIntiB, rataIntiB, sdIntiB);

			//Ambil nilai rerata Inti
			rerataIntiR = rataIntiR.val[0];
			rerataIntiG = rataIntiG.val[0];
			rerataIntiB = rataIntiB.val[0];
			rerataInti = rataInti.val[0];
			nilaiSDInti = sdInti[0];

			//Ambil nilai rerata Plasma
			rerataPlasmaR = rataPlasmaR.val[0];
			rerataPlasmaG = rataPlasmaG.val[0];
			rerataPlasmaB = rataPlasmaB.val[0];
			rerataPlasma = rataPlasma.val[0];
			nilaiSDPlasma = sdPlasma[0];

			//rerata = Rerata(citraModifikasiInti);
			//nilaiStddevInti = StdDev(citraModifikasiInti, rerata);
			circularityPlasma = Circularity(luasPlasma, kelilingPlasma);
			circularityInti = Circularity(luasInti, kelilingInti);
			liLP = PerbandinganIntiPlasma(luasInti, luasPlasma);
			kiKP = PerbandinganIntiPlasma(kelilingInti, kelilingPlasma);

			/*Tambahan Zharif*/
			luasNormalInti = LuasNormalisasiInti(luasInti, ambangIntiHasil.cols, ambangIntiHasil.rows);
			kelilingNormalInti = KelilingNormalisasiInti(kelilingInti, ambangIntiHasil.cols, ambangIntiHasil.rows);
			eccentricity = EccentricityInti(konturInti);

			float entropyInti, energyInti, contrastInti, homogenityInti, entropyPlasma, energyPlasma, contrastPlasma, homogenityPlasma;
			TeksturCitra(citraModIntiGray);
			entropyInti = entropi;
			energyInti = energi;
			contrastInti = kontras;
			homogenityInti = homogenitas;

			TeksturCitra(citraModPlasmaGray);
			entropyPlasma = entropi;
			energyPlasma = energi;
			contrastPlasma = kontras;
			homogenityPlasma = homogenitas;

			//Penulisan Teks
			textBoxGranularityPlasma->Text = nilaiSDPlasma.ToString();
			textBoxGranularityInti->Text = nilaiSDInti.ToString();
			textBoxCircularityPlasma->Text = circularityPlasma.ToString();
			textBoxCircularityInti->Text = circularityInti.ToString();
			textBoxLuasIntiPlasma->Text = liLP.ToString();
			textBoxKelilingIntiPlasma->Text = kiKP.ToString();
			textBoxLuasNormalisasiInti->Text = luasNormalInti.ToString();
			textBoxKelilingNormalisasiInti->Text = kelilingNormalInti.ToString();
			textBoxEccentricity->Text = eccentricity.ToString();

			textBoxRInti->Text = rerataIntiR.ToString();
			textBoxGInti->Text = rerataIntiG.ToString();
			textBoxBInti->Text = rerataIntiB.ToString();
			textBoxRPlasma->Text = rerataPlasmaR.ToString();
			textBoxGPlasma->Text = rerataPlasmaG.ToString();
			textBoxBPlasma->Text = rerataPlasmaB.ToString();

			textBoxEntropyInti->Text = entropyInti.ToString();
			textBoxEnergyInti->Text = energyInti.ToString();
			textBoxContrastInti->Text = contrastInti.ToString();
			textBoxHomogenityInti->Text = homogenityInti.ToString();
			textBoxEntropyPlasma->Text = entropyPlasma.ToString();
			textBoxEnergyPlasma->Text = energyPlasma.ToString();
			textBoxContrastPlasma->Text = contrastPlasma.ToString();
			textBoxHomogenityPlasma->Text = homogenityPlasma.ToString();

			//Jika mode olah banyak digunakan, jalankan ini
			if (mode == 1)
				fprintf(outfile, "%s\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", namaBerkasCitraFix,
					luasInti, kelilingInti, solidityInti, nilaiSDInti, circularityInti, rerataIntiR,
					rerataIntiG, rerataIntiB, entropyInti, energyInti, contrastInti, homogenityInti,
					luasPlasma, kelilingPlasma, solidityPlasma, nilaiSDPlasma, circularityPlasma, rerataPlasmaR,
					rerataPlasmaG, rerataPlasmaB, entropyPlasma, energyPlasma, contrastPlasma, homogenityPlasma,
					luasNormalInti, kelilingNormalInti, eccentricity, liLP, kiKP);

			//Radio button
			if (radioButtonBiner->Checked) {
				pictureBoxSitoplasma->Image = MatToBitmap(ambangPlasmaHasil);
				pictureBoxInti->Image = MatToBitmap(ambangIntiHasil);
			}
			else if (radioButtonKontur->Checked) {
				pictureBoxSitoplasma->Image = MatToBitmap(citraKonturPlasma);
				pictureBoxInti->Image = MatToBitmap(citraKonturInti);
			}
			else if (radioButtonConvexHull->Checked) {
				pictureBoxSitoplasma->Image = MatToBitmap(citraHullPlasma);
				pictureBoxInti->Image = MatToBitmap(citraHullInti);
			}
			else if (radioButtonConvexArea->Checked) {
				pictureBoxSitoplasma->Image = MatToBitmap(citraAreaHullPlasma);
				pictureBoxInti->Image = MatToBitmap(citraAreaHullInti);
			}
		}

		//Fungsi menghitung keliling
		int Keliling(Mat citraKontur) {
			int keliling = 0;
			int barisCitra = citraKontur.rows;
			int kolomCitra = *citraKontur.step.p;
			for (int x = 0; x < barisCitra; x++) {
				for (int y = 0; y < kolomCitra; y++) {
					if (citraKontur.at<uchar>(x, y) > 0) {
						keliling++;
					}
				}
			}
			return keliling;
		}
		//Fungsi menghitung luas convex
		int LuasConvex(Mat citraConvex) {
			int luas = 0;
			int barisCitra = citraConvex.rows;
			int kolomCitra = *citraConvex.step.p;
			for (int x = 0; x < barisCitra; x++) {
				for (int y = 0; y < kolomCitra; y++) {
					if (citraConvex.at<uchar>(x, y) > 0) {
						luas++;
					}
				}
			}
			return luas;
		}
		//Fungsi menghitung rata-rata
		float Rerata(Mat citraModif) {
			int n = 0;
			float rata = 0.00;
			float hasil = 0.00;
			for (int x = 0; x < citraModif.rows; x++) {
				for (int y = 0; y < citraModif.cols; y++) {
					if (citraModif.at<uchar>(x, y) > 0) {
						rata += citraModif.at<uchar>(x, y);
						n++;
					}
				}
			}
			hasil = rata / n;
			return (hasil);
		}
		//Fungsi menghitung standar deviasi
		float StdDev(Mat citraModif, float rata) {
			int n = 0;
			float nilaiStdDev = 0;
			for (int x = 0; x < citraModif.rows; x++) {
				for (int y = 0; y < citraModif.cols; y++) {
					if (citraModif.at<uchar>(x, y) > 0) {
						nilaiStdDev += pow(citraModif.at<uchar>(x, y) - rata, 2);
						n++;
					}
				}
			}
			return (sqrt(nilaiStdDev / (n - 1)));
		}
		//Fungsi menghitung tingkat kebulatan (circularity)
		float Circularity(int luas, int keliling) {
			float circularity = 4 * PI * luas / pow(keliling, 2);
			return circularity;
		}
		//Fungsi menghitung perbandingan fitur inti dan sitoplasma
		float PerbandinganIntiPlasma(int nilaiInti, int nilaiPlasma) {
			float perbandinganIntiPlasma = (Convert::ToDouble(nilaiInti)) / (Convert::ToDouble(nilaiPlasma));
			return perbandinganIntiPlasma;
		}

		/*Tambahan Zharif*/

		//Fungsi menghitung area normalisasi inti
		float LuasNormalisasiInti(int luasInti, int panjangInti, int lebarInti) {
			float luasNormalisasiInti = (Convert::ToDouble(luasInti)) /
				((Convert::ToDouble(panjangInti)*(Convert::ToDouble(lebarInti))));
			return luasNormalisasiInti;
		}
		//Fungsi menghitung keliling normalisasi inti
		float KelilingNormalisasiInti(int kelilingInti, int panjangInti, int lebarInti) {
			float kelilingNormalisasiInti = (Convert::ToDouble(kelilingInti)) /
				(2 * (Convert::ToDouble(panjangInti) + (Convert::ToDouble(lebarInti))));
			return kelilingNormalisasiInti;
		}
		//Fungsi menghitung eccentricity inti
		//Masih mbuh, mungkin bisa salah atau bener
		//vector<cv::Point>
		double EccentricityInti(vector<vector<cv::Point>> contours) {
			vector<Moments> mu(contours.size());
			int largestContourIndex;
			double largestArea;
			float myu20, myu11, myu02, eigenValue1, eigenValue2, eccentricityInti;
			Mat_<float> Matriks(2, 2);
			Mat eigenv, eigenvct;
			largestArea = 0;
			largestContourIndex = 0;
			myu20 = 0.00;
			myu11 = 0.00;
			myu02 = 0.00;

			for (int i = 0; i < contours.size(); i++) {
				double area = contourArea(contours[i], false);  //  Find the area of contour
				if (area > largestArea) {
					largestArea = area;
					largestContourIndex = i;
				}
			}

			mu[0] = moments(contours[largestContourIndex], false);
			myu20 = myu20 + mu[0].mu20;
			myu11 = myu11 + mu[0].mu11;
			myu02 = myu02 + mu[0].mu02;

			//Input nilai matriks ke dalam variabel
			//[myu20	myu11]
			//[myu11	myu02]
			Matriks(0, 0) = myu20;
			Matriks(1, 0) = myu11;
			Matriks(0, 1) = myu11;
			Matriks(1, 1) = myu02;

			//Hitung nilai eigen
			eigen(Matriks, eigenv, eigenvct);
			eigenValue1 = eigenv.at<float>(0, 0);
			eigenValue2 = eigenv.at<float>(1, 0);

			//Perhitungan eccentricity
			if (eigenValue1 >= eigenValue2)
				eccentricityInti = eigenValue2 / eigenValue1;
			else
				eccentricityInti = eigenValue1 / eigenValue2;

			return eccentricityInti;
		}

		//Fungsi menghitung tekstur citra
		void TeksturCitra(Mat CitraAbu) {
			float energy = 0, contrast = 0, homogenity = 0, IDM = 0, entropy = 0, mean1 = 0, tekstur[6];
			//array <float, 6> tekstur;
			int row = CitraAbu.rows, col = CitraAbu.cols;
			Mat gl = Mat::zeros(256, 256, CV_32FC1);

			//creating glcm matrix with 256 levels,radius=1 and in the horizontal direction 
			for (int i = 0; i < row; i++)
				for (int j = 0; j < col - 1; j++)
					gl.at<float>(CitraAbu.at<uchar>(i, j), CitraAbu.at<uchar>(i, j + 1)) = gl.at<float>(CitraAbu.at<uchar>(i, j), CitraAbu.at<uchar>(i, j + 1)) + 1;

			// normalizing glcm matrix for parameter determination
			gl = gl + gl.t();
			gl = gl / sum(gl)[0];


			for (int i = 0; i < 256; i++)
				for (int j = 0; j < 256; j++)
				{
					energy = energy + gl.at<float>(i, j)*gl.at<float>(i, j);            //finding parameters
					contrast = contrast + (i - j)*(i - j)*gl.at<float>(i, j);
					homogenity = homogenity + gl.at<float>(i, j) / (1 + abs(i - j));
					if (i != j)
						IDM = IDM + gl.at<float>(i, j) / ((i - j)*(i - j));                      //Taking k=2;
					if (gl.at<float>(i, j) != 0)
						entropy = entropy - gl.at<float>(i, j)*log10(gl.at<float>(i, j));
					mean1 = mean1 + 0.5*(i*gl.at<float>(i, j) + j*gl.at<float>(i, j));
				}
			entropi = entropy;
			energi = energy;
			kontras = contrast;
			homogenitas = homogenity;
		}

	private: System::Void checkBoxUjiPelatihan_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		statusUji = checkBoxUjiPelatihan->Checked;
		if (statusUji) {
			buttonBukaDataLatih->Text = "Model Latih";
			buttonMulaiPelatihan->Text = "Mulai Uji";
			label36->Visible = false;
			labelErrorDataTes->Visible = false;
		}
		else {
			buttonBukaDataLatih->Text = "Data latih";
			buttonMulaiPelatihan->Text = "Mulai Latih";
			label36->Visible = true;
			labelErrorDataTes->Visible = true;
		}
	}
	private: System::Void comboBoxActivationFunc_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		indexFAktif = comboBoxActivationFunc->SelectedIndex;
	}
	private: System::Void comboBoxTrainMethod_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		indexTrain = comboBoxTrainMethod->SelectedIndex;
		if (indexTrain == 0) {
			labelMomentum->Visible = true;
			labelWeight->Visible = true;
			labelDW0->Visible = false;
			labelDWMin->Visible = false;
			textBoxMomentum->Visible = true;
			textBoxWeight->Visible = true;
			textBoxDW0->Visible = false;
			textBoxDWMin->Visible = false;
		}
		else {
			labelMomentum->Visible = false;
			labelWeight->Visible = false;
			labelDW0->Visible = true;
			labelDWMin->Visible = true;
			textBoxMomentum->Visible = false;
			textBoxWeight->Visible = false;
			textBoxDW0->Visible = true;
			textBoxDWMin->Visible = true;
		}
	}
	private: System::Void numericUpDownBykHdn_ValueChanged(System::Object^  sender, System::EventArgs^  e) {
		bykHdnLayer = (int)numericUpDownBykHdn->Value;
		switch (bykHdnLayer) {
		case 1:
			labelHdn2->Visible = false;
			labelHdn3->Visible = false;
			labelHdn4->Visible = false;
			labelHdn5->Visible = false;
			numericUpDownHdn2->Visible = false;
			numericUpDownHdn3->Visible = false;
			numericUpDownHdn4->Visible = false;
			numericUpDownHdn5->Visible = false;
			numericUpDownHdn2->Value = 0;
			numericUpDownHdn3->Value = 0;
			numericUpDownHdn4->Value = 0;
			numericUpDownHdn5->Value = 0;
			hdnLayer2 = 0;
			hdnLayer3 = 0;
			hdnLayer4 = 0;
			hdnLayer5 = 0;
			break;
		case 2:
			labelHdn2->Visible = true;
			labelHdn3->Visible = false;
			labelHdn4->Visible = false;
			labelHdn5->Visible = false;
			numericUpDownHdn2->Visible = true;
			numericUpDownHdn3->Visible = false;
			numericUpDownHdn4->Visible = false;
			numericUpDownHdn5->Visible = false;
			numericUpDownHdn2->Value = 0;
			numericUpDownHdn3->Value = 0;
			numericUpDownHdn4->Value = 0;
			numericUpDownHdn5->Value = 0;
			hdnLayer2 = 0;
			hdnLayer3 = 0;
			hdnLayer4 = 0;
			hdnLayer5 = 0;
			break;
		case 3:
			labelHdn2->Visible = true;
			labelHdn3->Visible = true;
			labelHdn4->Visible = false;
			labelHdn5->Visible = false;
			numericUpDownHdn2->Visible = true;
			numericUpDownHdn3->Visible = true;
			numericUpDownHdn4->Visible = false;
			numericUpDownHdn5->Visible = false;
			numericUpDownHdn3->Value = 0;
			numericUpDownHdn4->Value = 0;
			numericUpDownHdn5->Value = 0;
			hdnLayer3 = 0;
			hdnLayer4 = 0;
			hdnLayer5 = 0;
			break;
		case 4:
			labelHdn2->Visible = true;
			labelHdn3->Visible = true;
			labelHdn4->Visible = true;
			labelHdn5->Visible = false;
			numericUpDownHdn2->Visible = true;
			numericUpDownHdn3->Visible = true;
			numericUpDownHdn4->Visible = true;
			numericUpDownHdn5->Visible = false;
			numericUpDownHdn4->Value = 0;
			numericUpDownHdn5->Value = 0;
			hdnLayer4 = 0;
			hdnLayer5 = 0;
			break;
		case 5:
			labelHdn2->Visible = true;
			labelHdn3->Visible = true;
			labelHdn4->Visible = true;
			labelHdn5->Visible = true;
			numericUpDownHdn2->Visible = true;
			numericUpDownHdn3->Visible = true;
			numericUpDownHdn4->Visible = true;
			numericUpDownHdn5->Visible = true;
			numericUpDownHdn5->Value = 0;
			hdnLayer5 = 0;
			break;
		}
	}
	private: System::Void numericUpDownHdn5_ValueChanged(System::Object^  sender, System::EventArgs^  e) {
		hdnLayer5 = (int)numericUpDownHdn5->Value;
	}
	private: System::Void numericUpDownHdn4_ValueChanged(System::Object^  sender, System::EventArgs^  e) {
		hdnLayer4 = (int)numericUpDownHdn4->Value;
	}
	private: System::Void numericUpDownHdn3_ValueChanged(System::Object^  sender, System::EventArgs^  e) {
		hdnLayer3 = (int)numericUpDownHdn3->Value;
	}
	private: System::Void numericUpDownHdn2_ValueChanged(System::Object^  sender, System::EventArgs^  e) {
		hdnLayer2 = (int)numericUpDownHdn2->Value;
	}
	private: System::Void numericUpDownOutput_ValueChanged(System::Object^  sender, System::EventArgs^  e) {
		outputLayer = (int)numericUpDownOutput->Value;
	}
	private: System::Void numericUpDownHdn1_ValueChanged(System::Object^  sender, System::EventArgs^  e) {
		hdnLayer1 = (int)numericUpDownHdn1->Value;
	}
	private: System::Void numericUpDownInputLayer_ValueChanged(System::Object^  sender, System::EventArgs^  e) {
		inputLayer = (int)numericUpDownInputLayer->Value;
	}
	private: System::Void buttonMulaiPelatihan_Click(System::Object^  sender, System::EventArgs^  e) {
		if (statusInputLatih && statusInputUji) {
			Mat layersKonfig, konfigurasi, errorHasil;
			layersKonfig = Mat(1, 8, CV_32F);
			konfigurasi = Mat(1, 10, CV_32F);
			errorHasil = Mat(2, 1, CV_32F);

			inputLayer = (int)numericUpDownInputLayer->Value;
			outputLayer = (int)numericUpDownOutput->Value;
			hdnLayer1 = (int)numericUpDownHdn1->Value;
			hdnLayer2 = (int)numericUpDownHdn2->Value;
			hdnLayer3 = (int)numericUpDownHdn3->Value;
			hdnLayer4 = (int)numericUpDownHdn4->Value;
			hdnLayer5 = (int)numericUpDownHdn5->Value;
			bykHdnLayer = (int)numericUpDownBykHdn->Value;

			layersKonfig.at<float>(0, 0) = inputLayer;
			layersKonfig.at<float>(0, 1) = outputLayer;
			layersKonfig.at<float>(0, 2) = hdnLayer1;
			layersKonfig.at<float>(0, 3) = hdnLayer2;
			layersKonfig.at<float>(0, 4) = hdnLayer3;
			layersKonfig.at<float>(0, 5) = hdnLayer4;
			layersKonfig.at<float>(0, 6) = hdnLayer5;
			layersKonfig.at<float>(0, 7) = bykHdnLayer;

			epsilon = Convert::ToDouble(textBoxEpsilon->Text);

			konfigurasi.at<float>(0, 0) = Convert::ToInt16(textBoxIterasiMLP->Text);
			konfigurasi.at<float>(0, 1) = Convert::ToDouble(textBoxEpsilon->Text);
			konfigurasi.at<float>(0, 2) = indexTrain;
			konfigurasi.at<float>(0, 3) = Convert::ToDouble(textBoxWeight->Text);
			konfigurasi.at<float>(0, 4) = Convert::ToDouble(textBoxMomentum->Text);
			konfigurasi.at<float>(0, 5) = Convert::ToDouble(textBoxDW0->Text);
			konfigurasi.at<float>(0, 6) = Convert::ToDouble(textBoxDWMin->Text);
			konfigurasi.at<float>(0, 7) = indexFAktif;
			konfigurasi.at<float>(0, 8) = Convert::ToDouble(textBoxAlpha->Text);
			konfigurasi.at<float>(0, 9) = Convert::ToDouble(textBoxBeta->Text);

			errorHasil = mlp(statusUji, dataLatih, dataUji, layersKonfig, konfigurasi, modelUji); //This will take long lol
			if (!statusUji)
				labelErrorDataTes->Text = Convert::ToString(errorHasil.at<float>(0, 0));
			labelErrorDataUji->Text = Convert::ToString(errorHasil.at<float>(1, 0));
		}
		else
			MessageBox::Show("Tolong lengkapi langkah 2 terlebih dahulu",
				"Kurang Data", MessageBoxButtons::OK,
				MessageBoxIcon::Warning);
	}
	private: System::Void radioButtonDTree_Click(System::Object^  sender, System::EventArgs^  e) {

	}
	private: System::Void radioButtonBayes_Click(System::Object^  sender, System::EventArgs^  e) {
	}
	private: System::Void radioButtonKNN_Click(System::Object^  sender, System::EventArgs^  e) {
	}
	private: System::Void radioButtonMLP_Click(System::Object^  sender, System::EventArgs^  e) {
		groupBoxSVM->Visible = false;
		groupBoxMLP->Visible = true;
	}
	private: System::Void radioButtonLatihSVM_Click(System::Object^  sender, System::EventArgs^  e) {
		groupBoxSVM->Visible = true;
		groupBoxMLP->Visible = false;
	}
	private: System::Void buttonBukaDataUji_Click(System::Object^  sender, System::EventArgs^  e) {
		if (openFileDialogTeks->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			int banyakVariabelInput, banyakVariabelOutput;
			cv::String berkasDataUji;
			System::String^ namaBerkasDataUji;
			berkasDataUji = MarshalString(openFileDialogTeks->FileName);
			namaBerkasDataUji = System::IO::Path::GetFileNameWithoutExtension(openFileDialogTeks->FileName);
			banyakVariabelInput = Convert::ToInt16(textBoxBanyakInput->Text);
			banyakVariabelOutput = Convert::ToInt16(textBoxBanyakOutput->Text);
			labelDataUji->Text = namaBerkasDataUji;
			dataUji = BacaDataTeks(berkasDataUji, banyakVariabelInput, banyakVariabelOutput);
			statusInputUji = true;
		}
	}
	private: System::Void buttonBukaDataLatih_Click(System::Object^  sender, System::EventArgs^  e) {
		if (openFileDialogTeks->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			int banyakVariabelInput, banyakVariabelOutput;
			cv::String berkasDataLatih;
			System::String^ namaBerkasDataLatih;
			berkasDataLatih = MarshalString(openFileDialogTeks->FileName);
			namaBerkasDataLatih = System::IO::Path::GetFileNameWithoutExtension(openFileDialogTeks->FileName);

			if (statusUji) {
				modelUji = MarshalString(openFileDialogTeks->FileName);
			}
			else {
				banyakVariabelInput = Convert::ToInt16(textBoxBanyakInput->Text);
				banyakVariabelOutput = Convert::ToInt16(textBoxBanyakOutput->Text);
				dataLatih = BacaDataTeks(berkasDataLatih, banyakVariabelInput, banyakVariabelOutput);
			}

			labelDataLatih->Text = namaBerkasDataLatih;
			statusInputLatih = true;
		}
	}
	private: System::Void MyForm_Load(System::Object^  sender, System::EventArgs^  e) {
		radioButtonMLP->Checked = true;
		groupBoxMLP->Visible = true;
	}

			 //Interaksi Pada Tab Ekstraksi
	private: System::Void radioButtonBiner_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		ProsesCitra(0);
	}
	private: System::Void radioButtonKontur_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		ProsesCitra(0);
	}
	private: System::Void radioButtonConvexHull_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		ProsesCitra(0);
	}
	private: System::Void radioButtonConvexArea_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		ProsesCitra(0);
	}
	private: System::Void hScrollBarPlasmaH_Scroll(System::Object^  sender, System::Windows::Forms::ScrollEventArgs^  e) {
		if (citraAwal.data != 0) {
			textBoxPlasmaH->Text = Convert::ToString(hScrollBarPlasmaH->Value);
			tPlasmaH = tPlasmaH + (hScrollBarPlasmaH->Value - tPlasmaH);
			ProsesCitra(0);
		}
	}
	private: System::Void hScrollBarPlasmaS_Scroll(System::Object^  sender, System::Windows::Forms::ScrollEventArgs^  e) {
		if (citraAwal.data != 0) {
			textBoxPlasmaS->Text = Convert::ToString(hScrollBarPlasmaS->Value);
			tPlasmaS = tPlasmaS + (hScrollBarPlasmaS->Value - tPlasmaS);
			ProsesCitra(0);
		}
	}
	private: System::Void hScrollBarPlasmaV_Scroll(System::Object^  sender, System::Windows::Forms::ScrollEventArgs^  e) {
		if (citraAwal.data != 0) {
			textBoxPlasmaV->Text = Convert::ToString(hScrollBarPlasmaV->Value);
			tPlasmaV = tPlasmaV + (hScrollBarPlasmaV->Value - tPlasmaV);
			ProsesCitra(0);
		}
	}
	private: System::Void hScrollBarIntiS_Scroll(System::Object^  sender, System::Windows::Forms::ScrollEventArgs^  e) {
		if (citraAwal.data != 0) {
			textBoxIntiS->Text = Convert::ToString(hScrollBarIntiS->Value);
			tIntiS = tIntiS + (hScrollBarIntiS->Value - tIntiS);
			ProsesCitra(0);
		}
	}
	private: System::Void buttonMuatCitra_Click(System::Object^  sender, System::EventArgs^  e) {
		if (openFileDialogCitra->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			citraAwal = cv::imread(MarshalString(openFileDialogCitra->FileName));
			pictureBoxRGB->Image = MatToBitmap(citraAwal);
			ProsesCitra(0);
		}
	}
	private: System::Void buttonProsesBanyak_Click(System::Object^  sender, System::EventArgs^  e) {
		if (folderBrowserDialog->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			//toolStripStatusProgram->Text="Proses olah banyak sedang berlangsung. Harap tunggu...";
			cv::String alamatFolder;
			int banyakCitra = 0, iCitra;
			System::String^ alamatFoldera;
			cli::array<System::String^>^ berkas;
			System::String^ namaberkas;
			alamatFoldera = folderBrowserDialog->SelectedPath;
			alamatFolder = MarshalString(folderBrowserDialog->SelectedPath);
			banyakCitra = System::IO::Directory::GetFiles(alamatFoldera)->Length;
			berkas = System::IO::Directory::GetFiles(alamatFoldera, "*.jpg", SearchOption::TopDirectoryOnly);

			char namaOutfile[60];
			sprintf(namaOutfile, "Data Citra Sel Darah Putih.txt");
			outfile = fopen(namaOutfile, "w");
			fprintf(outfile, "No Citra\tLuas I\tKeliling I\tSolidity I\tGranularity I\tCircularity I\tRerata R I\tRerata G I\tRerata B I\tEntropi I\tEnergi I\tKontras I\tHomogenitas I\tLuas P\tKeliling P\tSolidity P\tGranularity P\tCircularity P\tRerata R P\tRerata G P\tRerata B P\tEntropi P\tEnergi P\tKontras P\tHomogenitas P\tLuas Normal I P\tKeliling Normal I\tEccentricity\tLI per LP \tKI per KP\n");

			for (iCitra = 0; iCitra < banyakCitra - 1; iCitra++) {
				citraAwal = imread(MarshalString(berkas[iCitra]));
				namaberkas = System::IO::Path::GetFileNameWithoutExtension(berkas[iCitra]);
				namaBerkasCitra = (char*)(void*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(namaberkas);
				sprintf(namaBerkasCitraFix, namaBerkasCitra);
				ProsesCitra(1);
			}
			fclose(outfile);
			//toolStripStatusProgram->Text="Hasil proses olah banyak telah tersimpan.";
		}
	}
	private: System::Void muatCitraEkstraksiToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		if (openFileDialogCitra->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			citraAwal = cv::imread(MarshalString(openFileDialogCitra->FileName));
			pictureBoxRGB->Image = MatToBitmap(citraAwal);
			ProsesCitra(0);
		}
	}

	private: System::Void olahBanyakEkstraksiToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		if (folderBrowserDialog->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			//toolStripStatusProgram->Text="Proses olah banyak sedang berlangsung. Harap tunggu...";
			cv::String alamatFolder;
			int banyakCitra = 0, iCitra;
			System::String^ alamatFoldera;
			cli::array<System::String^>^ berkas;
			System::String^ namaberkas;
			alamatFoldera = folderBrowserDialog->SelectedPath;
			alamatFolder = MarshalString(folderBrowserDialog->SelectedPath);
			banyakCitra = System::IO::Directory::GetFiles(alamatFoldera)->Length;
			berkas = System::IO::Directory::GetFiles(alamatFoldera, "*.jpg", SearchOption::TopDirectoryOnly);

			char namaOutfile[60];
			sprintf(namaOutfile, "Data Citra Sel Darah Putih.txt");
			outfile = fopen(namaOutfile, "w");
			fprintf(outfile, "No Citra\tLuas I\tKeliling I\tSolidity I\tGranularity I\tCircularity I\tRerata R I\tRerata G I\tRerata B I\tEntropi I\tEnergi I\tKontras I\tHomogenitas I\tLuas P\tKeliling P\tSolidity P\tGranularity P\tCircularity P\tRerata R P\tRerata G P\tRerata B P\tEntropi P\tEnergi P\tKontras P\tHomogenitas P\tLuas Normal I P\tKeliling Normal I\tEccentricity\tLI per LP \tKI per KP\n");

			for (iCitra = 0; iCitra < banyakCitra - 1; iCitra++) {
				citraAwal = imread(MarshalString(berkas[iCitra]));
				namaberkas = System::IO::Path::GetFileNameWithoutExtension(berkas[iCitra]);
				namaBerkasCitra = (char*)(void*)System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(namaberkas);
				sprintf(namaBerkasCitraFix, namaBerkasCitra);
				ProsesCitra(1);
			}
			fclose(outfile);
			//toolStripStatusProgram->Text="Hasil proses olah banyak telah tersimpan.";
		}
	}
	};
}
