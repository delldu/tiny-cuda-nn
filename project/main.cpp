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
#include <string.h>
#include <stdlib.h>

#define CheckPoint(fmt, args...) printf("CheckPoint: %d(%s), " fmt "\n", __LINE__, __FILE__, ##args)


typedef struct applet_s {
	char *name;
	int (*main) (int argc, char **argv);
} applet_t;

#include "applets.h"

#define VERSION "1.0.0"
#define NUMBER_OF_APPLETS (int)(sizeof(applet_table)/sizeof(applet_table[0]))

static int applet_cmp(const void *a, const void *b)
{
	applet_t *m, *n;
	m = (applet_t *) a;
	n = (applet_t *) b;

	return strcmp(m->name, n->name);
}

applet_t *applet_search(char *name)
{
	applet_t key;
	key.name = name;

	return (applet_t *)bsearch(&key, applet_table, NUMBER_OF_APPLETS, sizeof(applet_t), applet_cmp);
}

extern void exit(int status);

void help(char *cmd)
{
	int i;
	printf("%s is a binary which has %d applets, version: %s.\n", cmd, NUMBER_OF_APPLETS, VERSION);
	printf("Usage: %s applet-name\n", cmd);
	printf("Support %d applets:\n", NUMBER_OF_APPLETS);
	for (i = 0; i < NUMBER_OF_APPLETS; i++)
		printf("  %s", help_table[i].name);
	printf("\n");

	exit(-1);
}

int main(int argc, char **argv)
{
	int cnt;
	char origname[256];
	char *p, *appletname, *rootname;
	applet_t *applet;

	rootname = argv[0];

	if ((cnt = readlink(argv[0], origname, sizeof(origname) - 1)) < 0 && strstr(argv[0], "meshbox")) {
		/*
		 * Guess: it must be box self !
		 * We must use the following command to run:
		 * nerfbox applet-name ......
		 */
		if (argc < 2)
			help(argv[0]);

		--argc;
		argv++;
	}

	appletname = argv[0];
	p = strrchr(appletname, '/');
	if (p)
		p++;
	else
		p = appletname;
	appletname = p;

	applet = applet_search(appletname);
	if (applet) {
		if (applet->main)
			return applet->main(argc, argv);
		else
			printf("Fatal error: applet->main is NULL !\n");
	} else {
		help(rootname);
	}

	return -1;
}
