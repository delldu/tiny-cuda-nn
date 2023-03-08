
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

#include "include/meshbox.h"


void eval_help(char *cmd)
{
	printf("Usage: %s [option]\n", cmd);
	printf("    -h, --help            Display this help.\n");
	printf("    -i, --input           Input directory (including image/depth/camera files).\n");

	exit(1);
}

int eval_main(int argc, char **argv)
{
	int optc;
	int option_index = 0;
	char *input_dir = NULL;

	struct option long_opts[] = {
		{ "help", 0, 0, 'h'},
		{"input", 1, 0, 'i'},
		{ 0, 0, 0, 0}

	};

	if (argc <= 1)
		eval_help(argv[0]);
	
	while ((optc = getopt_long(argc, argv, "h i:", long_opts, &option_index)) != EOF) {
		switch (optc) {
		case 'i':	// input
			input_dir = optarg;
			break;
		case 'h':	// help
		default:
			eval_help(argv[0]);
			break;
	    }
	}

	if (input_dir) {
		return eval_points(input_dir);
	}

	return 0;
}
