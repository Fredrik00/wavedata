import numpy as np

from wavedata.tools.core import geometry_utils


class VoxelGrid2D(object):
    """
    Voxel grids represent occupancy info. The voxelize_2d method projects a point cloud
    onto a plane, while saving height and point density information for each voxel.
    """

    # Class Constants
    VOXEL_EMPTY = -1
    VOXEL_FILLED = 0

    def __init__(self):

        # Quantization size of the voxel grid
        self.voxel_size = 0.0

        # Voxels at the most negative/positive xyz
        self.min_voxel_coord = np.array([])
        self.max_voxel_coord = np.array([])

        # Size of the voxel grid along each axis
        self.num_divisions = np.array([0, 0, 0])

        # Points in sorted order, to match the order of the voxels
        self.points = []

        # Indices of filled voxels
        self.voxel_indices = []

        # Max point height in projected voxel
        self.heights = []

        # Min point height in projected voxel
        self.min_heights = []

        # Number of points corresponding to projected voxel
        self.num_pts_in_voxel = []

        # Full occupancy grid, VOXEL_EMPTY or VOXEL_FILLED
        self.leaf_layout_2d = []

        # List of distances of filled voxels from origin, defaulted to a static value of 1
        self.distances = 1

        # Color of each point for visualization
        self.colors = []


    def voxelize_2d(self, pts, voxel_size, extents=None,
                    ground_plane=None, create_leaf_layout=True, maps=[]):
        """Voxelizes the point cloud into a 2D voxel grid by
        projecting it down into a flat plane, and stores the maximum
        point height, and number of points corresponding to the voxel

        :param pts: Point cloud as N x [x, y, z]
        :param voxel_size: Quantization size for the grid
        :param extents: Optional, specifies the full extents of the point cloud.
                        Used for creating same sized voxel grids.
        :param ground_plane: Plane coefficients (a, b, c, d), xz plane used if
                             not specified
        :param create_leaf_layout: Set this to False to create an empty
                                   leaf_layout, which will save computation
                                   time.
        """

        if len(pts) == 0:
            pts = np.array([[0,0,0]])  # Add single point in origin to prevent crashing

        # Check if points are 3D, otherwise early exit
        if pts.shape[1] != 3:
            raise ValueError("Points have the wrong shape: {}".format(
                pts.shape))

        self.voxel_size = voxel_size

        # Discretize voxel coordinates to given quantization size
        discrete_pts = np.floor(pts / voxel_size).astype(np.int32)

        # Use Lex Sort, sort by x, then z, then y (
        x_col = discrete_pts[:, 0]
        y_col = discrete_pts[:, 1]
        z_col = discrete_pts[:, 2]
        sorted_order = np.lexsort((y_col, z_col, x_col))

        # Save original points in sorted order
        self.points = pts[sorted_order]

        # Save discrete points in sorted order
        discrete_pts = discrete_pts[sorted_order]

        # Project all points to a 2D plane
        discrete_pts_2d = discrete_pts.copy()
        discrete_pts_2d[:, 1] = 0

        # Format the array to c-contiguous array for unique function
        contiguous_array = np.ascontiguousarray(discrete_pts_2d).view(
            np.dtype((np.void, discrete_pts_2d.dtype.itemsize *
                      discrete_pts_2d.shape[1])))

        # The new coordinates are the discretized array with its unique indexes
        _, unique_indices = np.unique(contiguous_array, return_index=True)

        # Sort unique indices to preserve order
        unique_indices.sort()

        if "max" in maps or "cluster" in maps:
            height_in_voxel = self.get_voxel_height(ground_plane, unique_indices)
            # Store the heights
            self.heights = height_in_voxel

        if "min" in maps or "cluster" in maps:
            # Returns the indices of the lowest coordinate by reading the reversed view
            _, unique_min_indices = np.unique(contiguous_array[::-1], return_index=True)

            # Reverse indices so they refer to the same point
            unique_min_indices = (len(contiguous_array) - 1) - unique_min_indices

            # Sort unique indices to preserve order
            unique_min_indices.sort()

            min_height_in_voxel = self.get_voxel_height(ground_plane, unique_min_indices)

            # Store the heights
            # NOTE min height can be larger than max height if difference is
            # less than the voxel size
            self.min_heights = min_height_in_voxel

        voxel_coords = discrete_pts_2d[unique_indices]

        # Number of points per voxel, last voxel calculated separately
        num_points_in_voxel = np.diff(unique_indices)
        num_points_in_voxel = np.append(num_points_in_voxel,
                                        discrete_pts_2d.shape[0] -
                                        unique_indices[-1])

        # Store number of points per voxel
        self.num_pts_in_voxel = num_points_in_voxel

        if "dnd" in maps:
            # Calculate distances decomposed from x and z coordinates for all filled voxels
            distances = np.sqrt(np.sum(np.square(voxel_coords*voxel_size), axis=1))
            self.distances = distances

        if "variance" in maps:
            # Probably incredibly slow...
            variance = np.zeros_like(num_points_in_voxel)
            j = 0
            for i in range(len(variance)):
                variance[i] = np.var(self.points[j:j+num_points_in_voxel[i], 1])
                j += num_points_in_voxel[i]

            # Store the height variance per voxel
            self.variance = variance

        if "cluster" in maps:
            global_clusters = []
            height_diffs = np.abs(self.heights - self.min_heights)
            avg_dists = height_diffs/num_points_in_voxel
            dists = np.abs(np.diff(self.points[:, 1]))
            for i in range(len(unique_indices)):
                first = unique_indices[i]
                local_clusters = [[first]]  # List of clusters with index of contained points
                longest_cluster = [first]
                longest = 1
                num_points = num_points_in_voxel[i]
                height_diff = height_diffs[i]
                average_distance = avg_dists[i]
                if average_distance < voxel_size:
                    average_distance = voxel_size

                for j in range(first, first + num_points - 1):
                    distance = dists[j]
                    if distance <= average_distance:
                        local_clusters[-1].append(j+1)  # Add to current cluster
                        if len(local_clusters[-1]) > longest:
                            longest_cluster = local_clusters[-1]
                            longest = len(longest_cluster)
                        
                    else:
                        local_clusters.append([j+1])  # Add new cluster

                #if num_points > 1 and height_diff > voxel_size/10: #and longest > 1: # Removes some noise, but potentially also objects
                global_clusters.append(longest_cluster)

            cluster_indices = np.array([cluster[0] for cluster in global_clusters]) # Mark cluster location by index of first point
            cluster_min_indices = np.array([cluster[-1] for cluster in global_clusters]) # Mark cluster end location by index of first point
            cluster_coords = discrete_pts_2d[cluster_indices]
            
            self.num_pts_in_cluster = np.array([len(cluster) for cluster in global_clusters])
            self.cluster_heights = self.get_voxel_height(ground_plane, cluster_indices)  # Take top point of clusters as max heights
            self.cluster_min_heights = self.get_voxel_height(ground_plane, cluster_min_indices)  # Take bottom point of clusters as min heights

            # In order to only draw selected clusters
            #self.colors = np.array([[0, 0, 0] for point in self.points])
            #for cluster in global_clusters:
            #   self.colors[cluster] = [1, 1, 1]
            #print(np.sum(np.sum(global_clusters)))

        # Find the minimum and maximum voxel coordinates
        if extents is not None:
            # Check provided extents
            extents_transpose = np.array(extents).transpose()
            if extents_transpose.shape != (2, 3):
                raise ValueError("Extents are the wrong shape {}".format(
                    extents.shape))

            # Set voxel grid extents
            self.min_voxel_coord = np.floor(extents_transpose[0]/voxel_size)
            self.max_voxel_coord = np.ceil((extents_transpose[1]/voxel_size)-1)

            self.min_voxel_coord[1] = 0
            self.max_voxel_coord[1] = 0

            # Check that points are bounded by new extents
            if not (self.min_voxel_coord <= np.amin(voxel_coords,
                                                    axis=0)).all():
                raise ValueError("Extents are smaller than min_voxel_coord")
            if not (self.max_voxel_coord >= np.amax(voxel_coords,
                                                    axis=0)).all():
                raise ValueError("Extents are smaller than max_voxel_coord")

        else:
            # Automatically calculate extents
            self.min_voxel_coord = np.amin(voxel_coords, axis=0)
            self.max_voxel_coord = np.amax(voxel_coords, axis=0)

        # Get the voxel grid dimensions
        self.num_divisions = ((self.max_voxel_coord - self.min_voxel_coord)
                              + 1).astype(np.int32)

        # Bring the min voxel to the origin
        self.voxel_indices = (voxel_coords - self.min_voxel_coord).astype(int)
        if "cluster" in maps:
            self.cluster_voxel_indices = (cluster_coords - self.min_voxel_coord).astype(int)

        if create_leaf_layout:
            # Create Voxel Object with -1 as empty/occluded, 0 as occupied
            self.leaf_layout_2d = self.VOXEL_EMPTY * \
                np.ones(self.num_divisions.astype(int))

            # Fill out the leaf layout
            self.leaf_layout_2d[self.voxel_indices[:, 0], 0,
                                self.voxel_indices[:, 2]] = \
                self.VOXEL_FILLED


    def map_to_index(self, map_index):
        """Converts map coordinate values to 1-based discretized grid index
        coordinate. Note: Any values outside the extent of the grid will be
        forced to be the maximum grid coordinate.

        :param map_index: N x 2 points

        :return: N x length(dim) (grid coordinate)
            [] if min_voxel_coord or voxel_size or grid_index or dim is not set
        """
        if self.voxel_size == 0 \
                or len(self.min_voxel_coord) == 0 \
                or len(map_index) == 0:
            return []

        num_divisions_2d = self.num_divisions[[0, 2]]
        min_voxel_coord_2d = self.min_voxel_coord[[0, 2]]

        # Truncate index (same as np.floor for positive values) and clip
        # to valid voxel index range
        indices = np.int32(map_index / self.voxel_size) - min_voxel_coord_2d
        indices[:, 0] = np.clip(indices[:, 0], 0, num_divisions_2d[0])
        indices[:, 1] = np.clip(indices[:, 1], 0, num_divisions_2d[1])

        return indices


    def get_voxel_height(self, ground_plane, indices):
            if ground_plane is None:
                # Use first point in voxel as highest point
                return self.points[indices, 1]
            else:
                # Ground plane provided
                return geometry_utils.dist_to_plane(ground_plane, self.points[indices])

