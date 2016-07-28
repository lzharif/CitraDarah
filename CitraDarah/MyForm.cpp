#include "MyForm.h"

using namespace CitraDarah;	// sesuai nama proyek

[STAThreadAttribute]
int main(cli::array<System::String ^> ^args)
{
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);
	// buat form yang dirancang dan jalankan
	Application::Run(gcnew MyForm());
	return 0;
}