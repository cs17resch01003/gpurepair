"""Module defining the error codes used by GPUVerify and gvtester."""

class ErrorCodes(object):
  SUCCESS = 0
  COMMAND_LINE_ERROR = 1
  CLANG_ERROR = 2
  OPT_ERROR = 3
  BUGLE_ERROR = 4
  GPUVERIFYVCGEN_ERROR = 5
  NOT_ALL_VERIFIED = 6
  TIMEOUT = 7
  CTRL_C = 8
  CONFIGURATION_ERROR = 9
  JSON_ERROR = 10
  BOOGIE_INTERNAL_ERROR = 11 # Internal failure of Boogie Driver or Cruncher
  BOOGIE_OTHER_ERROR = 12 # Uncategorised failure of Boogie Driver or Cruncher
  # The following is only used by gvtester.
  REGEX_MISMATCH_ERROR = 100
  # The following is only used for GPURepair
  INSTRUMENTATION_ERROR = 200
  ASSERTION_ERROR = 201
  REPAIR_ERROR = 202
  NONBARRIER_ERROR = 203
  SUMMARY_ERROR = 204
