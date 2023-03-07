
/************************************************************************************
***
***	Copyright 2023 Dell Du(18588220928@163.com), All Rights Reserved.
***
***	File Author: Dell, 2023年 03月 07日 星期二 18:29:34 CST
***
************************************************************************************/



#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>

#define DEFAULT_OUTPUT "output"


void fine_help(char *cmd)
{
	printf("Usage: %s [option]\n", cmd);
	printf("    -h, --help            Display this help.\n");
	printf("    -i, --input           Input directory (including mesh.obj and pc.ply).\n");
	printf("    -o, --output          Output directory, default: %s.\n", DEFAULT_OUTPUT);

	exit(1);
}

int fine_main(int argc, char **argv)
{
	int optc;
	int option_index = 0;
	char *input_dir = NULL;
	char *output_dir = (char *) DEFAULT_OUTPUT;

	struct option long_opts[] = {
		{ "help", 0, 0, 'h'},
		{"input", 1, 0, 'i'},
		{"output", 1, 0, 'o'},
		{ 0, 0, 0, 0}
	};

	if (argc <= 1)
		fine_help(argv[0]);
	
	while ((optc = getopt_long(argc, argv, "h i: o:", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'i':	// input
			input_dir = optarg;
			break;
		case 'o':	// output
			output_dir = optarg;
			break;
		case 'h':	// help
		default:
			fine_help(argv[0]);
			break;
	    }
	}

	if (input_dir == NULL) {
		fine_help(argv[0]);
	} else {
		printf("Training %s to model %s ...\n", input_dir, output_dir);
	}

	// MS -- Modify Section ?

	return 0;
}
