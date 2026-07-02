"""
    plot_timeseries_comparison(obs_dict, model_dict; kwargs...)

Plot observed and model time series with optional metrics annotation.

# Arguments
- `obs_dict`: Dict of `label => (time_vector, value_vector)` for observations
- `model_dict`: Dict of `label => (time_vector, value_vector)` for models

# Keyword Arguments
- `variable="Variable"`: Y-axis label
- `title=""`: plot title
- `show_metrics=true`: annotate RMSE/r for each model vs obs pair
- `figsize=(900, 400)`

# Returns
`Figure`
"""
function ValTools.plot_timeseries_comparison(
        obs_dict::Dict,
        model_dict::Dict;
        variable::String="Variable",
        title::String="",
        show_metrics::Bool=true,
        figsize::Tuple{Int,Int}=(900, 400))

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1]; ylabel=variable, xlabel="Time")

    obs_colors = [:black, :gray50, :saddlebrown, :navy]
    model_colors = Makie.wong_colors()

    # Plot observations (solid lines)
    for (i, (lbl, tv)) in enumerate(obs_dict)
        t, v = tv
        c = obs_colors[mod1(i, length(obs_colors))]
        lines!(ax, Float64.(1:length(t)), Float64.(v);
               color=c, linewidth=1.5, label=lbl)
    end

    # Plot models (dashed lines)
    metric_texts = String[]
    for (i, (lbl, tv)) in enumerate(model_dict)
        t, v = tv
        c = model_colors[mod1(i, length(model_colors))]
        lines!(ax, Float64.(1:length(t)), Float64.(v);
               color=c, linewidth=1.5, linestyle=:dash, label=lbl)

        if show_metrics
            for (obs_lbl, otv) in obs_dict
                _, vo = otv
                n = min(length(v), length(vo))
                if n > 3
                    m = ValTools.compute_metrics(Float64.(vo[1:n]), Float64.(v[1:n]))
                    push!(metric_texts,
                          "$lbl vs $obs_lbl: RMSE=$(round(m["rmse"]; digits=3)), r=$(round(m["correlation"]; digits=3))")
                end
            end
        end
    end

    axislegend(ax; position=:lt)

    if !isempty(metric_texts)
        txt = join(metric_texts, "\n")
        text!(ax, 0.02, 0.02; text=txt, fontsize=9, space=:relative,
              align=(:left, :bottom),
              color=:grey30)
    end

    !isempty(title) && (ax.title = title)

    return fig
end
