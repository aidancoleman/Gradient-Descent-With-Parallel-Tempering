MPICC  ?= mpicc
CFLAGS ?= -O3 -march=native -std=c11 -fopenmp -Wall -Wextra
LDLIBS ?= -lm

# Headers
DEPS = forest_model.h load_csv.h load_bin_mpi.h

# Objects shared by both mains
OBJS_COMMON = forest_model.o load_csv.o load_bin_mpi.o

all: train_pt_mpi

#weak-scaling main
train_pt_mpi: train_pt_mpi.o $(OBJS_COMMON)
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Strong-scaling main (alternate entry point)
train_pt_mpi_strong: fix_window_strong.o $(OBJS_COMMON)
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Convenience: build strong variant under the expected name
strong: train_pt_mpi_strong
	cp -f train_pt_mpi_strong train_pt_mpi

# Object rule; headers are dependencies
%.o: %.c $(DEPS)
	$(MPICC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o train_pt_mpi train_pt_mpi_strong *.log *.csv *.tmp

.PHONY: all clean strong
