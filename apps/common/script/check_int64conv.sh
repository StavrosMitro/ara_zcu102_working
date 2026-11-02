#!/bin/sh
# Fail the build if an ELF contains 64-bit int <-> float conversions.
#
# On the current ZCU102 bitstream fcvt.s.l / fcvt.s.lu / fcvt.l.s / fcvt.lu.s
# DO NOT READ THEIR OPERAND. With RVD=0 the scalar FLen is 32, so fpnew masks
# INT64 out of IntFmtConfig (fpnew_pkg.sv:441-456 pairs every int format with an
# equal-width FP format, so INT64 requires FP64) and the cast unit then reads the
# INACTIVE format's '{default: DONT_CARE} = all ones (fpnew_cast_multi.sv:239-245).
#   fcvt.s.lu  -> 0x4F800000 (2^32) for ANY input
#   fcvt.s.l   -> 0xBF800000 (-1.0) for ANY input
# There is no trap and no flag. See AGENT_NOTES_FP16_4LANE.md 43.3 and 46.8,
# where this silently zeroed fc_layer16only's weight gradients on every step.
#
# Usage: check_int64conv.sh <objdump> <elf>
# NB uses `-d`, never `-D`: -D disassembles embedded data (e.g. CIFAR arrays)
# as instructions and invents matches.
set -e
OBJDUMP="$1"
ELF="$2"
PAT='fcvt\.[sdh]\.lu?[[:space:]]|fcvt\.lu?\.[sdh][[:space:]]'

HITS=$("$OBJDUMP" -d "$ELF" 2>/dev/null | grep -E "$PAT" || true)
[ -z "$HITS" ] && exit 0

N=$(printf '%s\n' "$HITS" | wc -l)
echo ""
echo "*** BUILD BLOCKED: $ELF contains $N int64<->float conversion(s)."
echo "*** These return a CONSTANT on this hardware, silently and with no trap."
echo "***"
printf '%s\n' "$HITS" | head -10 | sed 's/^/***   /'
echo "***"
echo "*** Fix: narrow the operand to 32 bits at the conversion, e.g."
echo "***     (float)(uint32_t)n      instead of  (float)n"
echo "***     (_Float16)(uint32_t)n   instead of  (_Float16)n"
echo "*** where n is long / unsigned long / size_t / int64_t / a pointer diff."
echo "*** fcvt.s.w / fcvt.s.wu (INT32) are verified correct on this hardware."
echo "*** If the app deliberately tests this defect, add it to INT64CONV_EXEMPT"
echo "*** in apps/Makefile. See AGENT_NOTES_FP16_4LANE.md 43.3 / 46.8."
echo ""
exit 1
