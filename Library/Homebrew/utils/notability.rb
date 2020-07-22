# frozen_string_literal: true

require "utils/curl"

module SharedAudits
  module_function

  def github_repo_data(user, repo)
    @github_repo_data ||= {}
    @github_repo_data["#{user}/#{repo}"] ||= GitHub.repository(user, repo)

    @github_repo_data["#{user}/#{repo}"]
  rescue GitHub::HTTPNotFoundError
    nil
  end

  def gitlab_repo_data(user, repo)
    @gitlab_repo_data ||= {}
    @gitlab_repo_data["#{user}/#{repo}"] ||= begin
      out, _, status= curl_output("--request", "GET", "https://gitlab.com/api/v4/projects/#{user}%2F#{repo}")
      return unless status.success?

      JSON.parse(out)
    end

    @gitlab_repo_data["#{user}/#{repo}"]
  end

  def github(user, repo)
    metadata = github_repo_data(user, repo)

    return if metadata.nil?

    return "GitHub fork (not canonical repository)" if metadata["fork"]
    if (metadata["forks_count"] < 30) && (metadata["subscribers_count"] < 30) &&
       (metadata["stargazers_count"] < 75)
      return "GitHub repository not notable enough (<30 forks, <30 watchers and <75 stars)"
    end

    return if Date.parse(metadata["created_at"]) <= (Date.today - 30)

    "GitHub repository too new (<30 days old)"
  end

  def gitlab(user, repo)
    metadata = gitlab_repo_data(user, repo)

    return if metadata.nil?

    return "GitLab fork (not canonical repository)" if metadata["fork"]
    if (metadata["forks_count"] < 30) && (metadata["star_count"] < 75)
      return "GitLab repository not notable enough (<30 forks and <75 stars)"
    end

    return if Date.parse(metadata["created_at"]) <= (Date.today - 30)

    "GitLab repository too new (<30 days old)"
  end

  def bitbucket(user, repo)
    api_url = "https://api.bitbucket.org/2.0/repositories/#{user}/#{repo}"
    out, _, status= curl_output("--request", "GET", api_url)
    return unless status.success?

    metadata = JSON.parse(out)
    return if metadata.nil?

    return "Uses deprecated mercurial support in Bitbucket" if metadata["scm"] == "hg"

    return "Bitbucket fork (not canonical repository)" unless metadata["parent"].nil?

    return "Bitbucket repository too new (<30 days old)" if Date.parse(metadata["created_on"]) >= (Date.today - 30)

    forks_out, _, forks_status= curl_output("--request", "GET", "#{api_url}/forks")
    return unless forks_status.success?

    watcher_out, _, watcher_status= curl_output("--request", "GET", "#{api_url}/watchers")
    return unless watcher_status.success?

    forks_metadata = JSON.parse(forks_out)
    return if forks_metadata.nil?

    watcher_metadata = JSON.parse(watcher_out)
    return if watcher_metadata.nil?

    return if (forks_metadata["size"] < 30) && (watcher_metadata["size"] < 75)

    "Bitbucket repository not notable enough (<30 forks and <75 watchers)"
  end
end
