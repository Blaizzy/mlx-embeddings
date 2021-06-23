# typed: false
# frozen_string_literal: true

require "uri"
require "utils/github/actions"
require "utils/github/api"

# Wrapper functions for the GitHub API.
#
# @api private
module GitHub
  extend T::Sig

  module_function

  def check_runs(repo: nil, commit: nil, pr: nil)
    if pr
      repo = pr.fetch("base").fetch("repo").fetch("full_name")
      commit = pr.fetch("head").fetch("sha")
    end

    API.open_rest(url_to("repos", repo, "commits", commit, "check-runs"))
  end

  def create_check_run(repo:, data:)
    API.open_rest(url_to("repos", repo, "check-runs"), data: data)
  end

  def search_issues(query, **qualifiers)
    search("issues", query, **qualifiers)
  end

  def create_gist(files, description, private:)
    url = "#{API_URL}/gists"
    data = { "public" => !private, "files" => files, "description" => description }
    API.open_rest(url, data: data, scopes: CREATE_GIST_SCOPES)["html_url"]
  end

  def create_issue(repo, title, body)
    url = "#{API_URL}/repos/#{repo}/issues"
    data = { "title" => title, "body" => body }
    API.open_rest(url, data: data, scopes: CREATE_ISSUE_FORK_OR_PR_SCOPES)["html_url"]
  end

  def repository(user, repo)
    API.open_rest(url_to("repos", user, repo))
  end

  def search_code(repo: nil, user: "Homebrew", path: ["Formula", "Casks", "."], filename: nil, extension: "rb")
    matches = search("code", user: user, path: path, filename: filename, extension: extension, repo: repo)
    return matches if matches.blank?

    matches.map do |match|
      # .sub workaround for GitHub returning preceding /
      match["path"] = match["path"].delete_prefix("/")
      match
    end
  end

  def issues_for_formula(name, tap: CoreTap.instance, tap_remote_repo: tap.full_name, state: nil)
    search_issues(name, repo: tap_remote_repo, state: state, in: "title")
  end

  def user
    @user ||= API.open_rest("#{API_URL}/user")
  end

  def permission(repo, user)
    API.open_rest("#{API_URL}/repos/#{repo}/collaborators/#{user}/permission")
  end

  def write_access?(repo, user = nil)
    user ||= self.user["login"]
    ["admin", "write"].include?(permission(repo, user)["permission"])
  end

  def pull_requests(repo, **options)
    url = "#{API_URL}/repos/#{repo}/pulls?#{URI.encode_www_form(options)}"
    API.open_rest(url)
  end

  def merge_pull_request(repo, number:, sha:, merge_method:, commit_message: nil)
    url = "#{API_URL}/repos/#{repo}/pulls/#{number}/merge"
    data = { sha: sha, merge_method: merge_method }
    data[:commit_message] = commit_message if commit_message
    API.open_rest(url, data: data, request_method: :PUT, scopes: CREATE_ISSUE_FORK_OR_PR_SCOPES)
  end

  def print_pull_requests_matching(query, only = nil)
    open_or_closed_prs = search_issues(query, is: only, type: "pr", user: "Homebrew")

    open_prs, closed_prs = open_or_closed_prs.partition { |pr| pr["state"] == "open" }
                                             .map { |prs| prs.map { |pr| "#{pr["title"]} (#{pr["html_url"]})" } }

    if open_prs.present?
      ohai "Open pull requests"
      open_prs.each { |pr| puts pr }
    end

    if closed_prs.present?
      puts if open_prs.present?

      ohai "Closed pull requests"
      closed_prs.take(20).each { |pr| puts pr }

      puts "..." if closed_prs.count > 20
    end

    puts "No pull requests found for #{query.inspect}" if open_prs.blank? && closed_prs.blank?
  end

  def create_fork(repo, org: nil)
    url = "#{API_URL}/repos/#{repo}/forks"
    data = {}
    data[:organization] = org if org
    scopes = CREATE_ISSUE_FORK_OR_PR_SCOPES
    API.open_rest(url, data: data, scopes: scopes)
  end

  def check_fork_exists(repo, org: nil)
    _, reponame = repo.split("/")

    username = org || API.open_rest(url_to("user")) { |json| json["login"] }
    json = API.open_rest(url_to("repos", username, reponame))

    return false if json["message"] == "Not Found"

    true
  end

  def create_pull_request(repo, title, head, base, body)
    url = "#{API_URL}/repos/#{repo}/pulls"
    data = { title: title, head: head, base: base, body: body }
    scopes = CREATE_ISSUE_FORK_OR_PR_SCOPES
    API.open_rest(url, data: data, scopes: scopes)
  end

  def private_repo?(full_name)
    uri = url_to "repos", full_name
    API.open_rest(uri) { |json| json["private"] }
  end

  def query_string(*main_params, **qualifiers)
    params = main_params

    params += qualifiers.flat_map do |key, value|
      Array(value).map { |v| "#{key}:#{v}" }
    end

    "q=#{URI.encode_www_form_component(params.join(" "))}&per_page=100"
  end

  def url_to(*subroutes)
    URI.parse([API_URL, *subroutes].join("/"))
  end

  def search(entity, *queries, **qualifiers)
    uri = url_to "search", entity
    uri.query = query_string(*queries, **qualifiers)
    API.open_rest(uri) { |json| json.fetch("items", []) }
  end

  def approved_reviews(user, repo, pr, commit: nil)
    query = <<~EOS
      { repository(name: "#{repo}", owner: "#{user}") {
          pullRequest(number: #{pr}) {
            reviews(states: APPROVED, first: 100) {
              nodes {
                author {
                  ... on User { email login name databaseId }
                  ... on Organization { email login name databaseId }
                }
                authorAssociation
                commit { oid }
              }
            }
          }
        }
      }
    EOS

    result = API.open_graphql(query, scopes: ["user:email"])
    reviews = result["repository"]["pullRequest"]["reviews"]["nodes"]

    valid_associations = %w[MEMBER OWNER]
    reviews.map do |r|
      next if commit.present? && commit != r["commit"]["oid"]
      next unless valid_associations.include? r["authorAssociation"]

      email = r["author"]["email"].presence ||
              "#{r["author"]["databaseId"]}+#{r["author"]["login"]}@users.noreply.github.com"

      name = r["author"]["name"].presence ||
             r["author"]["login"]

      {
        "email" => email,
        "name"  => name,
        "login" => r["author"]["login"],
      }
    end.compact
  end

  def dispatch_event(user, repo, event, **payload)
    url = "#{API_URL}/repos/#{user}/#{repo}/dispatches"
    API.open_rest(url, data:           { event_type: event, client_payload: payload },
                       request_method: :POST,
                       scopes:         CREATE_ISSUE_FORK_OR_PR_SCOPES)
  end

  def workflow_dispatch_event(user, repo, workflow, ref, **inputs)
    url = "#{API_URL}/repos/#{user}/#{repo}/actions/workflows/#{workflow}/dispatches"
    API.open_rest(url, data:           { ref: ref, inputs: inputs },
                       request_method: :POST,
                       scopes:         CREATE_ISSUE_FORK_OR_PR_SCOPES)
  end

  def get_release(user, repo, tag)
    url = "#{API_URL}/repos/#{user}/#{repo}/releases/tags/#{tag}"
    API.open_rest(url, request_method: :GET)
  end

  def get_latest_release(user, repo)
    url = "#{API_URL}/repos/#{user}/#{repo}/releases/latest"
    API.open_rest(url, request_method: :GET)
  end

  def create_or_update_release(user, repo, tag, id: nil, name: nil, body: nil, draft: false)
    url = "#{API_URL}/repos/#{user}/#{repo}/releases"
    method = if id
      url += "/#{id}"
      :PATCH
    else
      :POST
    end
    data = {
      tag_name: tag,
      name:     name || tag,
      draft:    draft,
    }
    data[:body] = body if body.present?
    API.open_rest(url, data: data, request_method: method, scopes: CREATE_ISSUE_FORK_OR_PR_SCOPES)
  end

  def upload_release_asset(user, repo, id, local_file: nil, remote_file: nil)
    url = "https://uploads.github.com/repos/#{user}/#{repo}/releases/#{id}/assets"
    url += "?name=#{remote_file}" if remote_file
    API.open_rest(url, data_binary_path: local_file, request_method: :POST, scopes: CREATE_ISSUE_FORK_OR_PR_SCOPES)
  end

  def get_workflow_run(user, repo, pr, workflow_id: "tests.yml", artifact_name: "bottles")
    scopes = CREATE_ISSUE_FORK_OR_PR_SCOPES
    base_url = "#{API_URL}/repos/#{user}/#{repo}"
    pr_payload = API.open_rest("#{base_url}/pulls/#{pr}", scopes: scopes)
    pr_sha = pr_payload["head"]["sha"]
    pr_branch = URI.encode_www_form_component(pr_payload["head"]["ref"])
    parameters = "event=pull_request&branch=#{pr_branch}"

    workflow = API.open_rest("#{base_url}/actions/workflows/#{workflow_id}/runs?#{parameters}", scopes: scopes)
    workflow_run = workflow["workflow_runs"].select do |run|
      run["head_sha"] == pr_sha
    end

    [workflow_run, pr_sha, pr_branch, pr, workflow_id, scopes, artifact_name]
  end

  def get_artifact_url(workflow_array)
    workflow_run, pr_sha, pr_branch, pr, workflow_id, scopes, artifact_name = *workflow_array
    if workflow_run.empty?
      raise API::Error, <<~EOS
        No matching workflow run found for these criteria!
          Commit SHA:   #{pr_sha}
          Branch ref:   #{pr_branch}
          Pull request: #{pr}
          Workflow:     #{workflow_id}
      EOS
    end

    status = workflow_run.first["status"].sub("_", " ")
    if status != "completed"
      raise API::Error, <<~EOS
        The newest workflow run for ##{pr} is still #{status}!
          #{Formatter.url workflow_run.first["html_url"]}
      EOS
    end

    artifacts = API.open_rest(workflow_run.first["artifacts_url"], scopes: scopes)

    artifact = artifacts["artifacts"].select do |art|
      art["name"] == artifact_name
    end

    if artifact.empty?
      raise API::Error, <<~EOS
        No artifact with the name `#{artifact_name}` was found!
          #{Formatter.url workflow_run.first["html_url"]}
      EOS
    end

    artifact.first["archive_download_url"]
  end

  def public_member_usernames(org, per_page: 100)
    url = "#{API_URL}/orgs/#{org}/public_members"
    members = []

    API.paginate_rest(url, per_page: per_page) do |result|
      result = result.map { |member| member["login"] }
      members.concat(result)

      return members if result.length < per_page
    end
  end

  def members_by_team(org, team)
    query = <<~EOS
        { organization(login: "#{org}") {
          teams(first: 100) {
            nodes {
              ... on Team { name }
            }
          }
          team(slug: "#{team}") {
            members(first: 100) {
              nodes {
                ... on User { login name }
              }
            }
          }
        }
      }
    EOS
    result = API.open_graphql(query, scopes: ["read:org", "user"])

    if result["organization"]["teams"]["nodes"].blank?
      raise API::Error,
            "Your token needs the 'read:org' scope to access this API"
    end
    raise API::Error, "The team #{org}/#{team} does not exist" if result["organization"]["team"].blank?

    result["organization"]["team"]["members"]["nodes"].map { |member| [member["login"], member["name"]] }.to_h
  end

  def sponsors_by_tier(user)
    query = <<~EOS
        { organization(login: "#{user}") {
          sponsorsListing {
            tiers(first: 10, orderBy: {field: MONTHLY_PRICE_IN_CENTS, direction: DESC}) {
              nodes {
                monthlyPriceInDollars
                adminInfo {
                  sponsorships(first: 100, includePrivate: true) {
                    totalCount
                    nodes {
                      privacyLevel
                      sponsorEntity {
                        __typename
                        ... on Organization { login name }
                        ... on User { login name }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    EOS
    result = API.open_graphql(query, scopes: ["admin:org", "user"])

    tiers = result["organization"]["sponsorsListing"]["tiers"]["nodes"]

    tiers.map do |t|
      tier = t["monthlyPriceInDollars"]
      raise API::Error, "Your token needs the 'admin:org' scope to access this API" if t["adminInfo"].nil?

      sponsorships = t["adminInfo"]["sponsorships"]
      count = sponsorships["totalCount"]
      sponsors = sponsorships["nodes"].map do |sponsor|
        next unless sponsor["privacyLevel"] == "PUBLIC"

        se = sponsor["sponsorEntity"]
        {
          "name"  => se["name"].presence || sponsor["login"],
          "login" => se["login"],
          "type"  => se["__typename"].downcase,
        }
      end.compact

      {
        "tier"     => tier,
        "count"    => count,
        "sponsors" => sponsors,
      }
    end.compact
  end

  def get_repo_license(user, repo)
    response = API.open_rest("#{API_URL}/repos/#{user}/#{repo}/license")
    return unless response.key?("license")

    response["license"]["spdx_id"]
  rescue API::HTTPNotFoundError
    nil
  end

  def fetch_pull_requests(name, tap_remote_repo, state: nil, version: nil)
    if version.present?
      query = "#{name} #{version}"
      regex = /(^|\s)#{Regexp.quote(name)}(:|,|\s)(.*\s)?#{Regexp.quote(version)}(:|,|\s|$)/i
    else
      query = name
      regex = /(^|\s)#{Regexp.quote(name)}(:|,|\s|$)/i
    end
    issues_for_formula(query, tap_remote_repo: tap_remote_repo, state: state).select do |pr|
      pr["html_url"].include?("/pull/") && regex.match?(pr["title"])
    end
  rescue API::RateLimitExceededError => e
    opoo e.message
    []
  end

  def check_for_duplicate_pull_requests(name, tap_remote_repo, state:, file:, args:, version: nil)
    pull_requests = fetch_pull_requests(name, tap_remote_repo, state: state, version: version).select do |pr|
      pr_files = API.open_rest(url_to("repos", tap_remote_repo, "pulls", pr["number"], "files"))
      pr_files.any? { |f| f["filename"] == file }
    end
    return if pull_requests.blank?

    duplicates_message = <<~EOS
      These pull requests may be duplicates:
      #{pull_requests.map { |pr| "#{pr["title"]} #{pr["html_url"]}" }.join("\n")}
    EOS
    error_message = "Duplicate PRs should not be opened. Use --force to override this error."
    if args.force? && !args.quiet?
      opoo duplicates_message
    elsif !args.force? && args.quiet?
      odie error_message
    elsif !args.force?
      odie <<~EOS
        #{duplicates_message.chomp}
        #{error_message}
      EOS
    end
  end

  def forked_repo_info!(tap_remote_repo, org: nil)
    response = create_fork(tap_remote_repo, org: org)
    # GitHub API responds immediately but fork takes a few seconds to be ready.
    sleep 1 until check_fork_exists(tap_remote_repo, org: org)
    remote_url = if system("git", "config", "--local", "--get-regexp", "remote\..*\.url", "git@github.com:.*")
      response.fetch("ssh_url")
    else
      url = response.fetch("clone_url")
      if (api_token = Homebrew::EnvConfig.github_api_token)
        url.gsub!(%r{^https://github\.com/}, "https://#{api_token}@github.com/")
      end
      url
    end
    username = response.fetch("owner").fetch("login")
    [remote_url, username]
  end

  def create_bump_pr(info, args:)
    tap = info[:tap]
    sourcefile_path = info[:sourcefile_path]
    old_contents = info[:old_contents]
    additional_files = info[:additional_files] || []
    remote = info[:remote] || "origin"
    remote_branch = info[:remote_branch] || tap.path.git_origin_branch
    branch = info[:branch_name]
    commit_message = info[:commit_message]
    previous_branch = info[:previous_branch] || "-"
    tap_remote_repo = info[:tap_remote_repo] || tap.full_name
    pr_message = info[:pr_message]

    sourcefile_path.parent.cd do
      git_dir = Utils.popen_read("git", "rev-parse", "--git-dir").chomp
      shallow = !git_dir.empty? && File.exist?("#{git_dir}/shallow")
      changed_files = [sourcefile_path]
      changed_files += additional_files if additional_files.present?

      if args.dry_run? || (args.write? && !args.commit?)
        remote_url = if args.no_fork?
          Utils.popen_read("git", "remote", "get-url", "--push", "origin").chomp
        else
          fork_message = "try to fork repository with GitHub API" \
                         "#{" into `#{args.fork_org}` organization" if args.fork_org}"
          ohai fork_message
          "FORK_URL"
        end
        ohai "git fetch --unshallow origin" if shallow
        ohai "git add #{changed_files.join(" ")}"
        ohai "git checkout --no-track -b #{branch} #{remote}/#{remote_branch}"
        ohai "git commit --no-edit --verbose --message='#{commit_message}'" \
             " -- #{changed_files.join(" ")}"
        ohai "git push --set-upstream #{remote_url} #{branch}:#{branch}"
        ohai "git checkout --quiet #{previous_branch}"
        ohai "create pull request with GitHub API (base branch: #{remote_branch})"
      else

        unless args.commit?
          if args.no_fork?
            remote_url = Utils.popen_read("git", "remote", "get-url", "--push", "origin").chomp
            username = tap.user
          else
            begin
              remote_url, username = forked_repo_info!(tap_remote_repo, org: args.fork_org)
            rescue *API::ERRORS => e
              sourcefile_path.atomic_write(old_contents)
              odie "Unable to fork: #{e.message}!"
            end
          end

          safe_system "git", "fetch", "--unshallow", "origin" if shallow
        end

        safe_system "git", "add", *changed_files
        safe_system "git", "checkout", "--no-track", "-b", branch, "#{remote}/#{remote_branch}" unless args.commit?
        safe_system "git", "commit", "--no-edit", "--verbose",
                    "--message=#{commit_message}",
                    "--", *changed_files
        return if args.commit?

        safe_system "git", "push", "--set-upstream", remote_url, "#{branch}:#{branch}"
        safe_system "git", "checkout", "--quiet", previous_branch
        pr_message = <<~EOS
          #{pr_message}
        EOS
        user_message = args.message
        if user_message
          pr_message = <<~EOS
            #{user_message}

            ---

            #{pr_message}
          EOS
        end

        begin
          url = create_pull_request(tap_remote_repo, commit_message,
                                    "#{username}:#{branch}", remote_branch, pr_message)["html_url"]
          if args.no_browse?
            puts url
          else
            exec_browser url
          end
        rescue *API::ERRORS => e
          odie "Unable to open pull request: #{e.message}!"
        end
      end
    end
  end

  def pull_request_commits(user, repo, pr, per_page: 100)
    pr_data = API.open_rest(url_to("repos", user, repo, "pulls", pr))
    commits_api = pr_data["commits_url"]
    commit_count = pr_data["commits"]
    commits = []

    if commit_count > API_MAX_ITEMS
      raise API::Error, "Getting #{commit_count} commits would exceed limit of #{API_MAX_ITEMS} API items!"
    end

    API.paginate_rest(commits_api, per_page: per_page) do |result, page|
      commits.concat(result.map { |c| c["sha"] })

      return commits if commits.length == commit_count

      if result.empty? || page * per_page >= commit_count
        raise API::Error, "Expected #{commit_count} commits but actually got #{commits.length}!"
      end
    end
  end

  def pull_request_labels(user, repo, pr)
    pr_data = API.open_rest(url_to("repos", user, repo, "pulls", pr))
    pr_data["labels"].map { |label| label["name"] }
  end
end
