#include <stdio.h>
#include "FileUtils.cuh"

int n_lines(const char *file)
{

    FILE *myfile = fopen(file, "r");
    int ch, n_lines = 0;

    do {
        ch = fgetc(myfile);
        if (ch == '\n') {
            n_lines++;
        }
    }
    while (ch != EOF);

    // last line doesn't end with a new line!
    // but there has to be a line at least before the last line
    if (ch != '\n' && n_lines != 0) {
        n_lines++;
    }

    fclose(myfile);

    return n_lines;

}

bool file_exists(const char *filename)
{
    if (FILE *file = fopen(filename, "r")) {
        fclose(file);
        return true;
    }
    return false;
}
