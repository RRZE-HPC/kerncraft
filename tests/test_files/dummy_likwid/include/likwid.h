#include <stdio.h>

void likwid_markerInit(void) {
  fprintf(stderr, "WARNING: dummy likwid_markerInit() called\n");
};
void likwid_markerThreadInit(void) {
  fprintf(stderr, "WARNING: dummy likwid_markerThreadInit() called\n");
};
int likwid_markerStartRegion(const char* regionTag) {
  fprintf(stderr, "WARNING: dummy likwid_markerStartRegion(...) called\n");
  return 0;
};
int likwid_markerStopRegion(const char* regionTag) {
  fprintf(stderr, "WARNING: dummy likwid_markerStopRegion(...) called\n");
  return 0;};
void likwid_markerClose(void) {
  fprintf(stderr, "WARNING: dummy likwid_markerClose() called\n");
};
