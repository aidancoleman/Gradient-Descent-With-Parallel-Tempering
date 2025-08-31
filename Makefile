MPICC ?= mpicc
CFLAGS ?= -O3 -march=native -std=c11 -fopenmp -Wall -Wextra
LDFLAGS ?= -lm

#executables
EXE_STRONG := train_strongscale       # from learning_strongscale.c
EXE_WEAK := data_weak_scaling       # from data_weak_scaling.c
EXE_PT := train_pt_mpi            # from train_pt_mpi.c

# program-specific sources
SRC_STRONG := learning_strongscale.c
SRC_WEAK := data_weak_scaling.c
SRC_PT := train_pt_mpi.c

#common sources (provide the missing symbols)
SRC_COMMON := forest_model.c load_csv.c load_bin_mpi.c
OBJ_COMMON := $(SRC_COMMON:.c=.o)

# auto-deps
CFLAGS += -MMD -MP

.PHONY: all strong weak pt clean

all: strong weak pt
strong: $(EXE_STRONG)
weak: $(EXE_WEAK)
pt: $(EXE_PT)

# link rules and we reuse common objects so they compile once!
$(EXE_STRONG): $(SRC_STRONG) $(OBJ_COMMON)
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(EXE_WEAK): $(SRC_WEAK) $(OBJ_COMMON)
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(EXE_PT): $(SRC_PT) $(OBJ_COMMON)
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

#compile common objects
%.o: %.c
	$(MPICC) $(CFLAGS) -c $< -o $@

-include $(OBJ_COMMON:.o=.d)

clean:
	rm -f $(EXE_STRONG) $(EXE_WEAK) $(EXE_PT) $(OBJ_COMMON) $(OBJ_COMMON:.o=.d)
