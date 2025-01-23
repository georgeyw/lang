from typing import Dict, List, Union, Optional

import wandb
import pandas as pd


def get_run_by_name(
    entity: str,
    project: str,
    run_name: str
) -> str:
    """
    Get the run ID for a specific run name in a project.

    Args:
        entity (str): W&B username or team name
        project (str): Name of the W&B project
        run_name (str): Name of the run to find

    Returns:
        str: Run ID of the matching run
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", {"display_name": run_name})

    # Convert to list to check length
    runs_list = list(runs)

    if not runs_list:
        raise ValueError(f"No run found with name '{run_name}'")
    if len(runs_list) > 1:
        # If multiple runs found, warn user and return the most recent one
        print(f"Warning: Multiple runs found with name '{run_name}'. Using most recent.")

    # Return the ID of the first (most recent) run
    return runs_list[0].id

def get_wandb_run_data(
    entity: str,
    project: str,
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, Union[pd.DataFrame, dict]]:
    """
    Pull data from a specified W&B run.

    Args:
        entity (str): W&B username or team name
        project (str): Name of the W&B project
        run_id (str, optional): ID of the specific run to pull data from
        run_name (str, optional): Name of the run to pull data from (alternative to run_id)
        metrics (List[str], optional): List of specific metrics to pull. If None, pulls all metrics.

    Returns:
        Dict containing:
            - 'metrics': DataFrame of run metrics history
            - 'config': Dictionary of run configuration
            - 'summary': Dictionary of run summary statistics
    """
    if not run_id and not run_name:
        raise ValueError("Must provide either run_id or run_name")

    if run_name and not run_id:
        run_id = get_run_by_name(entity, project, run_name)

    try:
        # Initialize W&B API and get run
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")

        # Get metrics history
        metrics_df = pd.DataFrame(run.history())

        # Add step column if it doesn't exist
        if '_step' not in metrics_df.columns:
            metrics_df['_step'] = range(len(metrics_df))

        # Filter metrics if specified
        if metrics:
            # Always include _step column if it exists
            columns_to_keep = ['_step'] + metrics
            metrics_df = metrics_df[columns_to_keep]

        # Get run configuration and summary
        config_dict = dict(run.config)
        summary_dict = dict(run.summary._json_dict)

        return {
            'metrics': metrics_df,
            'config': config_dict,
            'summary': summary_dict
        }

    except Exception as e:
        raise RuntimeError(f"Error pulling W&B data: {e}") from e