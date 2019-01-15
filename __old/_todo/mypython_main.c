 
void mypython_main_interactive(int argc, char** argv)
{
  Py_Initialize();
  Py_Main(argc, argv);
  Py_Finalize();
}

void mypython_main_code(const char* code)
{
  Py_Initialize();
  PyRun_SimpleString(code);
  Py_Finalize();
}

//cc mypython.c -lpthread -ldl -lutil -lm -lpython2.5
//-lpython2.3 -lm -L/usr/lib/python2.3/config
