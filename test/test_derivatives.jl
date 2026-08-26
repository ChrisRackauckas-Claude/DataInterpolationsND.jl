using DataInterpolationsND
using ForwardDiff
using Random

function test_deriv_10(itp, t, t_ad = t)
    return @test itp(t; derivative_orders = (1, 0)) ≈
        ForwardDiff.derivative(t₁ -> itp(t₁, t_ad[2]), t_ad[1])
end

function test_deriv_01(itp, t, t_ad = t)
    return @test itp(t; derivative_orders = (0, 1)) ≈
        ForwardDiff.derivative(t₂ -> itp(t_ad[1], t₂), t_ad[2])
end

function test_deriv_11(itp, t, t_ad = t)
    return @test itp(t; derivative_orders = (1, 1)) ≈
        ForwardDiff.derivative(
        t₂ -> ForwardDiff.derivative(
            t₁ -> itp(t₁, t₂), t_ad[1]
        ),
        t_ad[2]
    )
end

function test_derivatives_between_data_points(itp::NDInterpolation{2})
    t1 = itp.interp_dims[1].t
    t2 = itp.interp_dims[2].t

    for t in Iterators.product(
            t1[1:(end - 1)] + diff(t1) / 2, t2[1:(end - 1)] + diff(t2) / 2
        )
        test_deriv_10(itp, t)
        test_deriv_01(itp, t)
        test_deriv_11(itp, t)
    end
    return
end

function test_derivatives_at_data_points(itp::NDInterpolation{2})
    t1 = itp.interp_dims[1].t
    t2 = itp.interp_dims[2].t
    for t in Iterators.product(t1, t2)
        test_deriv_10(itp, t)
        test_deriv_01(itp, t)
        test_deriv_11(itp, t)
    end
    return
end

function test_linear_derivatives_at_data_points(itp::NDInterpolation{2})
    t1 = itp.interp_dims[1].t
    t2 = itp.interp_dims[2].t
    t1_cell = [sum(t1[1:2]) / 2; t1[1:(end - 1)] + diff(t1) / 2]
    t2_cell = [sum(t2[1:2]) / 2; t2[1:(end - 1)] + diff(t2) / 2]

    # Linear derivatives are undefined at knots, so test the one-sided cell convention.
    for (t, t_cell) in zip(Iterators.product(t1, t2), Iterators.product(t1_cell, t2_cell))
        test_deriv_10(itp, t, (t_cell[1], t[2]))
        test_deriv_01(itp, t, (t[1], t_cell[2]))
        test_deriv_11(itp, t, t_cell)
    end
    return
end

@testset "Linear Interpolation" begin
    Random.seed!(1)
    t1 = [1.0, 2.0, 3.5, 4.0, 10.0]
    t2 = [-1.5, 0.0, 2.5, 7.8, 12.9]
    u = rand(5, 5)
    itp_dims = (
        LinearInterpolationDimension(t1),
        LinearInterpolationDimension(t2),
    )
    itp = NDInterpolation(u, itp_dims)
    test_linear_derivatives_at_data_points(itp)
    test_derivatives_between_data_points(itp)
end

@testset "BSpline interpolation" begin
    Random.seed!(1)
    t1 = [1.0, 2.0, 3.5, 4.0, 10.0]
    t2 = [-1.5, 0.0, 2.5, 7.8, 12.9]
    u = rand(6, 6)
    itp_dims = (
        BSplineInterpolationDimension(t1, 2),
        BSplineInterpolationDimension(t2, 2),
    )
    itp = NDInterpolation(u, itp_dims)
    test_derivatives_at_data_points(itp)
    test_derivatives_between_data_points(itp)

    u = rand(9, 9)
    itp_dims = (
        BSplineInterpolationDimension(t1, 5),
        BSplineInterpolationDimension(t2, 5),
    )
    itp = NDInterpolation(u, itp_dims)
    test_derivatives_at_data_points(itp)
    test_derivatives_between_data_points(itp)
end
