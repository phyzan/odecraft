#pragma once

#include <memory>
#include <ndspan/ndspan.hpp>
#include <odecraft/Toolkit/Scratch.hpp>

namespace ode::interp::sci {


template<size_t NDIM>
class DelaunayTri{

    static constexpr double EPS = std::numeric_limits<double>::epsilon() * 100;

    static constexpr ndspan::Allocation SPX_ALLOC = NDIM == 0 ? ndspan::Allocation::Heap : ndspan::Allocation::Stack;

public:

    static constexpr int DIM_SPX = (NDIM == 0 ? 0 : NDIM+1);

    DelaunayTri(const DelaunayTri& other) = delete;

    DelaunayTri& operator=(const DelaunayTri& other) = delete;

    DelaunayTri(DelaunayTri&& other) noexcept = default;

    DelaunayTri& operator=(DelaunayTri&& other) noexcept = default;

    DelaunayTri(const double* points, size_t n_points, size_t ndim);

    // Allowes derived classes to set a state into the object
    // wihtout having to recompute the triangulation.
    DelaunayTri() = default;

    void set_state(const ndspan::CView2D<const double, 0, NDIM>& points, const ndspan::CView2D<const int, 0, DIM_SPX>& simplices, const ndspan::CView2D<const int, 0, DIM_SPX>& neighbors, const ndspan::CView2D<const double, 0, NDIM>& v0, const ndspan::CView3D<const double, 0, NDIM, NDIM>& invT);

    // Member variable accessors
    const ndspan::CArray2D<double, 0, NDIM>& get_points() const; // (n_points, ndim)
    const ndspan::CArray2D<int, 0, DIM_SPX>& get_simplices() const; // (nsimplices, ndim+1)
    const ndspan::CArray2D<int, 0, DIM_SPX>& get_neighbors() const; // (nsimplices, ndim+1)
    const ndspan::CArray2D<double, 0, NDIM>& get_v0() const; // (nsimplices, ndim)
    const ndspan::CArray3D<double, 0, NDIM, NDIM>& get_invT() const; // (nsimplices, ndim, ndim)
    

    int                 ndim() const;
    int                 npoints() const;
    
    int                 nsimplices() const;
    
    const int*          get_simplex(int simplex_idx) const; //ndim + 1 elements
    const int*          get_neighbors(int simplex_idx) const; // ndim + 1 elements, -1 = no neighbor (boundary)
    // A DelaunayTri keeps no state between queries, so it may be shared amongst threads. Its
    // caller provides what a query works with: cached_simplex, the simplex to start the search
    // from, such as the one the previous search ended in (-1 to start anywhere), and worker, an
    // ndim x ndim scratch matrix. A TriView keeps both for its queries.
    int                 find_simplex(const double* point, int cached_simplex) const; // array of ndim elements, returns simplex index or -1 if not found
    bool                contains(const double* point, int cached_simplex) const { return find_simplex(point, cached_simplex) != -1; }
    // out: ndim + 1 elements, the barycentric coordinates of point in the simplex's vertices, in the
    // order of get_simplex. Returns false if any is not finite, as for a degenerate simplex
    bool                compute_barycentric(double* out, int simplex_idx, const double* point) const;

    // field must be a contiguous array of shape (npoints, n_fields), the n_fields values of each
    // point together, and the output of interpolate is a contiguous array of shape (n_fields,).
    // Returns the simplex the point is in, or -1 if it is outside the hull, with out set to NaN
    int                 interpolate(double* out, const double* point, const double* field, int n_fields, int cached_simplex) const;

    double              get_value(const double* point, const double* field, int cached_simplex) const; // field should have npoints elements, returns interpolated value at point

    double              volume(int simplex_idx, ndspan::CArray2D<double, NDIM, NDIM>& worker) const; // returns the volume of the simplex with index simplex_idx

    double              total_volume(ndspan::CArray2D<double, NDIM, NDIM>& worker) const;

    // auxiliary function for interpolation, used when the caller already has the barycentric coordinates and simplex index (e.g. for multiple fields)


private:

    void compute_delaunay_1d();

    // ============================================================================
    // Qhull-based Delaunay triangulation - O(n log n)
    // ============================================================================

    void compute_delaunay_nd();

    void compute_delaunay();

    ndspan::CArray2D<double, 0, NDIM>        points_;    // (npoints, ndim)
    ndspan::CArray2D<int, 0, DIM_SPX>        simplices_; // (nsimplices, ndim+1)
    ndspan::CArray2D<int, 0, DIM_SPX>        neighbors_; // (nsimplices, ndim+1), -1 = boundary
    ndspan::CArray2D<double, 0, NDIM>        v0_;        // (nsimplices, ndim)
    ndspan::CArray3D<double, 0, NDIM, NDIM>  invT_;      // (nsimplices, ndim, ndim)

};


/*
Instantiate a TriView and do not share it amongst threads.
It holds a shared pointer to a DelaunayTri.
For multithreading operations that require all threads to
use the same DelaunayTri, simply instantiate one or more
TriView's per thread.
*/
template<size_t NDIM>
class TriView{
public:
    TriView(std::shared_ptr<DelaunayTri<NDIM>> tri) : mat_scratch_(ndspan::CArray2D<double, NDIM, NDIM>(tri->ndim(), tri->ndim())), tri_(std::move(tri)) {}

    const std::shared_ptr<DelaunayTri<NDIM>>& operator->() const {
        return tri_;
    }

    std::shared_ptr<DelaunayTri<NDIM>>& operator->() {
        return tri_;
    }

    const std::shared_ptr<DelaunayTri<NDIM>>& shared_tri() const {
        return tri_;
    }

    double  volume(int simplex_idx) const {
        decltype(auto) worker = mat_scratch_.get();
        return tri_->volume(simplex_idx, worker);
    }

    double total_volume() const {
        decltype(auto) worker = mat_scratch_.get();
        return tri_->total_volume(worker);
    }

    // Each search starts from the simplex the last successful one of this view ended in
    int find_simplex(const double* point) const {
        return remember(tri_->find_simplex(point, cached_simplex_));
    }

    bool contains(const double* point) const {
        return find_simplex(point) != -1;
    }

    // As DelaunayTri::interpolate: returns the simplex the point is in, or -1 outside the hull
    int interpolate(double* out, const double* point, const double* field, int n_fields=1) const {
        return remember(tri_->interpolate(out, point, field, n_fields, cached_simplex_));
    }

    double get_value(const double* point, const double* field) const {
        double res;
        interpolate(&res, point, field, 1);
        return res;
    }

private:

    // A point outside the hull leaves the simplex to start the next search from as it was
    int remember(int simplex) const {
        if (simplex != -1){
            cached_simplex_ = simplex;
        }
        return simplex;
    }

    worker_t<ndspan::CArray2D<double, NDIM, NDIM>> mat_scratch_;
    std::shared_ptr<DelaunayTri<NDIM>> tri_;
    mutable int cached_simplex_ = -1;
};

} // namespace ode::interp::sci
