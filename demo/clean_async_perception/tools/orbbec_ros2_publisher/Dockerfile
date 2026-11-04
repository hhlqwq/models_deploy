FROM intel-perception:22.04 AS build
WORKDIR /build
COPY dep/linux-x86_64/orbbec_sdk /opt/orbbec_sdk
COPY tools/orbbec_ros2_publisher/CMakeLists.txt tools/orbbec_ros2_publisher/main.cpp /src/
RUN bash -lc 'source /opt/ros/humble/setup.bash && cmake -S /src -B /build && cmake --build /build --parallel'

FROM intel-perception:22.04
COPY --from=build /opt/orbbec_sdk/lib /opt/orbbec_publisher/lib
COPY --from=build /build/orbbec_ros2_publisher /opt/orbbec_publisher/bin/orbbec_ros2_publisher
ENV LD_LIBRARY_PATH=/opt/orbbec_publisher/lib
CMD ["/opt/orbbec_publisher/bin/orbbec_ros2_publisher"]
