program main
  use omp_lib
  implicit none
  integer :: GET_THREAD_NUM
  integer :: GET_NUM_THREADS
  integer :: GET_TEAM_NUM
  integer :: GET_NUM_TEAMS
  integer :: GET_LEVEL
  GET_THREAD_NUM = -1
  GET_NUM_THREADS =  -1
  GET_TEAM_NUM = -1
  GET_NUM_TEAMS =  -1
  GET_LEVEL =  -1
  !$omp target map(from:GET_THREAD_NUM,GET_NUM_THREADS, & 
  !$omp& GET_LEVEL,GET_TEAM_NUM,GET_NUM_TEAMS)
      GET_THREAD_NUM = omp_get_thread_num()
      GET_NUM_THREADS = omp_get_num_threads()
      GET_TEAM_NUM = omp_get_team_num()
      GET_NUM_TEAMS = omp_get_num_teams()
      GET_LEVEL = omp_is_initial_device()
  !$omp end target 
  write(*,*) "  omp_get_thread_num()  ", GET_THREAD_NUM
  write(*,*) "  omp_get_num_threads() ", GET_NUM_THREADS
  write(*,*) "  omp_get_team_num()    ", GET_TEAM_NUM
  write(*,*) "  omp_get_num_teams()   ", GET_NUM_TEAMS
  write(*,*) "  omp_get_level()       ", GET_LEVEL
end program main
