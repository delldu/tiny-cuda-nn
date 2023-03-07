
/* Warning: 
 * Don't edit this table manually. !!! 
 *
 */

extern int eval_main(int agrc, char **argv);
extern int fine_main(int agrc, char **argv);

static applet_t applet_table[] = {
	{ (char *)"eval", eval_main },
	{ (char *)"fine", fine_main },
};

static applet_t help_table[] = {
	{ (char *)"eval", eval_main },
	{ (char *)"fine", fine_main },
};

