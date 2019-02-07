#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include "../check_offloading/check.h"

int main(int argc, char *argv[]){

  if (check_offloading())
    return 1;

  //Open file
  std::ifstream infile(argv[1]);
  assert( infile.is_open() && "File does not exist!");

  // Go line by line
  std::string line;

  while(std::getline(infile,line)){
    printf(" ---> Testing %s...\n", line.c_str());

    std::string lib;
    lib += line;
    lib += "/results/a.so";

    void *h = dlopen(lib.c_str(),RTLD_NOW);

    assert(h && "Library not found!");

    // Prepare the function and arguments
    void *entry = NULL;
    if(!entry) entry = dlsym(h,"main");

    assert(entry && "Sym not found!");

    auto entryf = (int(*)(int,char*[]))entry;

    int largc = 1;
    char largv0[] = "test";
    char *largv[1] = {&largv0[0]};

    // Redirect stdout/err
    int old_stdout = dup(1);
    int old_stderr = dup(2);

    std::string name_stdout;
    name_stdout += line;
    name_stdout += "/results/stdout";
    std::string name_stderr;
    name_stderr += line;
    name_stderr += "/results/stderr";

    FILE *fout = fopen(name_stdout.c_str(),"w");
    FILE *ferr = fopen(name_stderr.c_str(),"w");
    assert(fout && ferr && "Can't open stderr/out files!");
    //assert( dup2(fileno(fout), 1) > 0 && "Redirect stdout failed!");
    //assert( dup2(fileno(ferr), 2) > 0 && "Redirect stderr failed!");

    // Run library
    printf(" ---> Before %s...\n", line.c_str());
    entryf(largc,largv);
    printf(" ---> After %s...\n", line.c_str());

    // Redirect things back
    //assert( dup2(old_stdout, 1) && "Inv redirect stdout failed!");
    //assert( dup2(old_stderr, 2) && "Inv redirect stdout failed!");
    fclose(fout);
    fclose(ferr);

    // Close the library
    dlclose(h);
  }

}
