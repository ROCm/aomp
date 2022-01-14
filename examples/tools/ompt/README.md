OMPT target support: Examples to demonstrate how a tool would use the OMPT target APIs
=======================================================================================

The examples simulate how a tool is expected to use OMPT target
support. The tool would register callbacks and call OMPT runtime entry
points to start and stop device tracing, if required. The tool would
have an OpenMP thread call these runtime entry points to control
device tracing. When certain events occur, the OpenMP runtime would
invoke the event callbacks so that the tool can establish the event
context. If device tracing has been requested, the OpenMP runtime
would collect and manage trace records in buffers. When a buffer fills
up or if an OpenMP thread requests explicit flushing of trace records,
an OpenMP runtime helper thread would invoke a buffer-completion
callback. The buffer-completion callback is implemented by the tool
and would typically traverse the trace records returned as part of the
callback. Once the trace records are returned, they can be correlated
to the context established earlier through the event callbacks.

Here are the steps:
(1) The tool has to define a function called ompt_start_tool with
C-linkage and the appropriate signature as defined by the OpenMP
spec. This function provides 2 function pointers as part of the
returned object, one for an initialization function and the other for
a finalization function.

(2) The tool has to define the initialization and the finalization
functions referred to above. The initialization function is invoked by
the OpenMP runtime with an input lookup parameter. Typically, the
initialization function would use the lookup parameter to obtain a
handle to the function ompt_set_callback that is implemented by the
OpenMP runtime. Using this handle, the tool can then register
callbacks. In our examples for OMPT target, some common callbacks
registered include device initialization, data transfer operations,
and target submit.

(3) The device initialize callback, implemented by the tool, is
invoked by the OpenMP device plugin runtime during device
initialization with a lookup parameter. This callback would look up
entry points (such as ompt_start_trace) for device tracing so that the
tool can control the regions that should be traced.

(4) The ompt_start_trace entry point expects 2 function pointers, one
for an allocation function that will be invoked by the OpenMP runtime
for allocating space for trace record buffers. The other one is a
buffer-completion callback function that will be invoked by an OpenMP
runtime helper thread for returning trace records to the tool. The
tool is expected to use the entry point, ompt_get_record_ompt, to
inspect a trace record at a given cursor and the entry point,
ompt_advance_buffer_cursor, to traverse the returned trace records.

(5) If device tracing is desired, calls to entry points,
ompt_set_trace_ompt, ompt_start_trace, ompt_flush_trace, and
ompt_stop_trace will be injected into the OpenMP program by the tool
to control the type and region of tracing.
