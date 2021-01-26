# typed: false
# frozen_string_literal: true

require "tempfile"
require "uri"
require "utils/github/actions"
require "utils/shell"

# Helper functions for interacting with the GitHub API.
#
# @api private
module GitHub
  extend T::Sig

  module_function

  API_URL = "https://api.github.com"
  API_MAX_PAGES = 50
  API_MAX_ITEMS = 5000

  CREATE_GIST_SCOPES = ["gist"].freeze
  CREATE_ISSUE_FORK_OR_PR_SCOPES = ["public_repo"].freeze
  ALL_SCOPES = (CREATE_GIST_SCOPES + CREATE_ISSUE_FORK_OR_PR_SCOPES).freeze
  ALL_SCOPES_URL = Formatter.url(
    "https://github.com/settings/tokens/new?scopes=#{ALL_SCOPES.join(",")}&description=Homebrew",
  ).freeze

  # Generic API error.
  class Error < RuntimeError
    attr_reader :github_message
  end

  # Error when the requested URL is not found.
  class HTTPNotFoundError < Error
    def initialize(github_message)
      @github_message = github_message
      super
    end
  end

  # Error when the API rate limit is exceeded.
  class RateLimitExceededError < Error
    def initialize(reset, github_message)
      @github_message = github_message
      super <<~EOS
        GitHub API Error: #{github_message}
        Try again in #{pretty_ratelimit_reset(reset)}, or create a personal access token:
          #{ALL_SCOPES_URL}
        #{Utils::Shell.set_variable_in_profile("HOMEBREW_GITHUB_API_TOKEN", "your_token_here")}
      EOS
    end

    def pretty_ratelimit_reset(reset)
      pretty_duration(Time.at(reset) - Time.now)
    end
  end

  # Error when authentication fails.
  class AuthenticationFailedError < Error
    def initialize(github_message)
      @github_message = github_message
      message = +"GitHub #{github_message}:"
      message << if Homebrew::EnvConfig.github_api_token
        <<~EOS
          HOMEBREW_GITHUB_API_TOKEN may be invalid or expired; check:
            #{Formatter.url("https://github.com/settings/tokens")}
        EOS
      else
        <<~EOS
          The GitHub credentials in the macOS keychain may be invalid.
          Clear them with:
            printf "protocol=https\\nhost=github.com\\n" | git credential-osxkeychain erase
          Or create a personal access token:
            #{ALL_SCOPES_URL}
          #{Utils::Shell.set_variable_in_profile("HOMEBREW_GITHUB_API_TOKEN", "your_token_here")}
        EOS
      end
      super message.freeze
    end
  end

  # Error when the API returns a validation error.
  class ValidationFailedError < Error
    def initialize(github_message, errors)
      @github_message = if errors.empty?
        github_message
      else
        "#{github_message}: #{errors}"
      end

      super(@github_message)
    end
  end

  API_ERRORS = [
    AuthenticationFailedError,
    HTTPNotFoundError,
    RateLimitExceededError,
    Error,
    JSON::ParserError,
  ].freeze

  # Gets the password field from `git-credential-osxkeychain` for github.com,
  # but only if that password looks like a GitHub Personal Access Token.
  sig { returns(T.nilable(String)) }
  def keychain_username_password
    github_credentials = Utils.popen(["git", "credential-osxkeychain", "get"], "w+") do |pipe|
      pipe.write "protocol=https\nhost=github.com\n"
      pipe.close_write
      pipe.read
    end
    github_username = github_credentials[/username=(.+)/, 1]
    github_password = github_credentials[/password=(.+)/, 1]
    return unless github_username

    # Don't use passwords from the keychain unless they look like
    # GitHub Personal Access Tokens:
    #   https://github.com/Homebrew/brew/issues/6862#issuecomment-572610344
    return unless /^[a-f0-9]{40}$/i.match?(github_password)

    github_password
  rescue Errno::EPIPE
    # The above invocation via `Utils.popen` can fail, causing the pipe to be
    # prematurely closed (before we can write to it) and thus resulting in a
    # broken pipe error. The root cause is usually a missing or malfunctioning
    # `git-credential-osxkeychain` helper.
    nil
  end

  def api_credentials
    @api_credentials ||= begin
      Homebrew::EnvConfig.github_api_token || keychain_username_password
    end
  end

  sig { returns(Symbol) }
  def api_credentials_type
    if Homebrew::EnvConfig.github_api_token
      :env_token
    elsif keychain_username_password
      :keychain_username_password
    else
      :none
    end
  end

  # Given an API response from GitHub, warn the user if their credentials
  # have insufficient permissions.
  def api_credentials_error_message(response_headers, needed_scopes)
    return if response_headers.empty?

    scopes = response_headers["x-accepted-oauth-scopes"].to_s.split(", ")
    return if scopes.present?

    needed_human_scopes = needed_scopes.join(", ")
    credentials_scopes = response_headers["x-oauth-scopes"]
    return if needed_human_scopes.blank? && credentials_scopes.blank?

    needed_human_scopes = "none" if needed_human_scopes.blank?
    credentials_scopes = "none" if credentials_scopes.blank?

    what = case api_credentials_type
    when :keychain_username_password
      "macOS keychain GitHub"
    when :env_token
      "HOMEBREW_GITHUB_API_TOKEN"
    end

    @api_credentials_error_message ||= onoe <<~EOS
      Your #{what} credentials do not have sufficient scope!
      Scopes required: #{needed_human_scopes}
      Scopes present:  #{credentials_scopes}
      Create a personal access token:
        #{ALL_SCOPES_URL}
      #{Utils::Shell.set_variable_in_profile("HOMEBREW_GITHUB_API_TOKEN", "your_token_here")}
    EOS
  end

  def open_api(url, data: nil, data_binary_path: nil, request_method: nil, scopes: [].freeze, parse_json: true)
    # This is a no-op if the user is opting out of using the GitHub API.
    return block_given? ? yield({}) : {} if Homebrew::EnvConfig.no_github_api?

    args = ["--header", "Accept: application/vnd.github.v3+json", "--write-out", "\n%\{http_code}"]
    args += ["--header", "Accept: application/vnd.github.antiope-preview+json"]

    token = api_credentials
    args += ["--header", "Authorization: token #{token}"] unless api_credentials_type == :none

    data_tmpfile = nil
    if data
      begin
        data = JSON.generate data
        data_tmpfile = Tempfile.new("github_api_post", HOMEBREW_TEMP)
      rescue JSON::ParserError => e
        raise Error, "Failed to parse JSON request:\n#{e.message}\n#{data}", e.backtrace
      end
    end

    if data_binary_path.present?
      args += ["--data-binary", "@#{data_binary_path}"]
      args += ["--header", "Content-Type: application/gzip"]
    end

    headers_tmpfile = Tempfile.new("github_api_headers", HOMEBREW_TEMP)
    begin
      if data
        data_tmpfile.write data
        data_tmpfile.close
        args += ["--data", "@#{data_tmpfile.path}"]

        args += ["--request", request_method.to_s] if request_method
      end

      args += ["--dump-header", headers_tmpfile.path]

      output, errors, status = curl_output("--location", url.to_s, *args, secrets: [token])
      output, _, http_code = output.rpartition("\n")
      output, _, http_code = output.rpartition("\n") if http_code == "000"
      headers = headers_tmpfile.read
    ensure
      if data_tmpfile
        data_tmpfile.close
        data_tmpfile.unlink
      end
      headers_tmpfile.close
      headers_tmpfile.unlink
    end

    begin
      raise_api_error(output, errors, http_code, headers, scopes) if !http_code.start_with?("2") || !status.success?

      return if http_code == "204" # No Content

      output = JSON.parse output if parse_json
      if block_given?
        yield output
      else
        output
      end
    rescue JSON::ParserError => e
      raise Error, "Failed to parse JSON response\n#{e.message}", e.backtrace
    end
  end

  def open_graphql(query, scopes: [].freeze)
    data = { query: query }
    result = open_api("https://api.github.com/graphql", scopes: scopes, data: data, request_method: "POST")

    raise Error, result["errors"].map { |e| "#{e["type"]}: #{e["message"]}" }.join("\n") if result["errors"].present?

    result["data"]
  end

  def raise_api_error(output, errors, http_code, headers, scopes)
    json = begin
      JSON.parse(output)
    rescue
      nil
    end
    message = json&.[]("message") || "curl failed! #{errors}"

    meta = {}
    headers.lines.each do |l|
      key, _, value = l.delete(":").partition(" ")
      key = key.downcase.strip
      next if key.empty?

      meta[key] = value.strip
    end

    if meta.fetch("x-ratelimit-remaining", 1).to_i <= 0
      reset = meta.fetch("x-ratelimit-reset").to_i
      raise RateLimitExceededError.new(reset, message)
    end

    api_credentials_error_message(meta, scopes)

    case http_code
    when "401", "403"
      raise AuthenticationFailedError, message
    when "404"
      raise HTTPNotFoundError, message
    when "422"
      errors = json&.[]("errors") || []
      raise ValidationFailedError.new(message, errors)
    else
      raise Error, message
    end
  end

  def check_runs(repo: nil, commit: nil, pr: nil)
    if pr
      repo = pr.fetch("base").fetch("repo").fetch("full_name")
      commit = pr.fetch("head").fetch("sha")
    end

    open_api(url_to("repos", repo, "commits", commit, "check-runs"))
  end

  def create_check_run(repo:, data:)
    open_api(url_to("repos", repo, "check-runs"), data: data)
  end

  def search_issues(query, **qualifiers)
    search("issues", query, **qualifiers)
  end

  def repository(user, repo)
    open_api(url_to("repos", user, repo))
  end

  def search_code(**qualifiers)
    matches = search("code", **qualifiers)
    return matches if matches.blank?

    matches.map do |match|
      # .sub workaround for GitHub returning preceding /
      match["path"] = match["path"].delete_prefix("/")
      match
    end
  end

  def issues_for_formula(name, tap: CoreTap.instance, tap_full_name: tap.full_name, state: nil)
    search_issues(name, repo: tap_full_name, state: state, in: "title")
  end

  def user
    @user ||= open_api("#{API_URL}/user")
  end

  def permission(repo, user)
    open_api("#{API_URL}/repos/#{repo}/collaborators/#{user}/permission")
  end

  def write_access?(repo, user = nil)
    user ||= self.user["login"]
    ["admin", "write"].include?(permission(repo, user)["permission"])
  end

  def pull_requests(repo, **options)
    url = "#{API_URL}/repos/#{repo}/pulls?#{URI.encode_www_form(options)}"
    open_api(url)
  end

  def merge_pull_request(repo, number:, sha:, merge_method:, commit_message: nil)
    url = "#{API_URL}/repos/#{repo}/pulls/#{number}/merge"
    data = { sha: sha, merge_method: merge_method }
    data[:commit_message] = commit_message if commit_message
    open_api(url, data: data, request_method: :PUT, scopes: CREATE_ISSUE_FORK_OR_PR_SCOPES)
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

  def create_fork(repo)
    url = "#{API_URL}/repos/#{repo}/forks"
    data = {}
    scopes = CREATE_ISSUE_FORK_OR_PR_SCOPES
    open_api(url, data: data, scopes: scopes)
  end

  def check_fork_exists(repo)
    _, reponame = repo.split("/")

    username = open_api(url_to("user")) { |json| json["login"] }
    json = open_api(url_to("repos", username, reponame))

    return false if json["message"] == "Not Found"

    true
  end

  def create_pull_request(repo, title, head, base, body)
    url = "#{API_URL}/repos/#{repo}/pulls"
    data = { title: title, head: head, base: base, body: body }
    scopes = CREATE_ISSUE_FORK_OR_PR_SCOPES
    open_api(url, data: data, scopes: scopes)
  end

  def private_repo?(full_name)
    uri = url_to "repos", full_name
    open_api(uri) { |json| json["private"] }
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
    open_api(uri) { |json| json.fetch("items", []) }
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

    result = open_graphql(query, scopes: ["user:email"])
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
    open_api(url, data:           { event_type: event, client_payload: payload },
                  request_method: :POST,
                  scopes:         CREATE_ISSUE_FORK_OR_PR_SCOPES)
  end

  def workflow_dispatch_event(user, repo, workflow, ref, **inputs)
    url = "#{API_URL}/repos/#{user}/#{repo}/actions/workflows/#{workflow}/dispatches"
    open_api(url, data:           { ref: ref, inputs: inputs },
                  request_method: :POST,
                  scopes:         CREATE_ISSUE_FORK_OR_PR_SCOPES)
  end

  def get_release(user, repo, tag)
    url = "#{API_URL}/repos/#{user}/#{repo}/releases/tags/#{tag}"
    open_api(url, request_method: :GET)
  end

  def get_latest_release(user, repo)
    url = "#{API_URL}/repos/#{user}/#{repo}/releases/latest"
    open_api(url, request_method: :GET)
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
    open_api(url, data: data, request_method: method, scopes: CREATE_ISSUE_FORK_OR_PR_SCOPES)
  end

  def upload_release_asset(user, repo, id, local_file: nil, remote_file: nil)
    url = "https://uploads.github.com/repos/#{user}/#{repo}/releases/#{id}/assets"
    url += "?name=#{remote_file}" if remote_file
    open_api(url, data_binary_path: local_file, request_method: :POST, scopes: CREATE_ISSUE_FORK_OR_PR_SCOPES)
  end

  def get_workflow_run(user, repo, pr, workflow_id: "tests.yml", artifact_name: "bottles")
    scopes = CREATE_ISSUE_FORK_OR_PR_SCOPES
    base_url = "#{API_URL}/repos/#{user}/#{repo}"
    pr_payload = open_api("#{base_url}/pulls/#{pr}", scopes: scopes)
    pr_sha = pr_payload["head"]["sha"]
    pr_branch = URI.encode_www_form_component(pr_payload["head"]["ref"])
    parameters = "event=pull_request&branch=#{pr_branch}"

    workflow = open_api("#{base_url}/actions/workflows/#{workflow_id}/runs?#{parameters}", scopes: scopes)
    workflow_run = workflow["workflow_runs"].select do |run|
      run["head_sha"] == pr_sha
    end

    [workflow_run, pr_sha, pr_branch, pr, workflow_id, scopes, artifact_name]
  end

  def get_artifact_url(workflow_array)
    workflow_run, pr_sha, pr_branch, pr, workflow_id, scopes, artifact_name = *workflow_array
    if workflow_run.empty?
      raise Error, <<~EOS
        No matching workflow run found for these criteria!
          Commit SHA:   #{pr_sha}
          Branch ref:   #{pr_branch}
          Pull request: #{pr}
          Workflow:     #{workflow_id}
      EOS
    end

    status = workflow_run.first["status"].sub("_", " ")
    if status != "completed"
      raise Error, <<~EOS
        The newest workflow run for ##{pr} is still #{status}!
          #{Formatter.url workflow_run.first["html_url"]}
      EOS
    end

    artifacts = open_api(workflow_run.first["artifacts_url"], scopes: scopes)

    artifact = artifacts["artifacts"].select do |art|
      art["name"] == artifact_name
    end

    if artifact.empty?
      raise Error, <<~EOS
        No artifact with the name `#{artifact_name}` was found!
          #{Formatter.url workflow_run.first["html_url"]}
      EOS
    end

    artifact.first["archive_download_url"]
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
    result = open_graphql(query, scopes: ["admin:org", "user"])

    tiers = result["organization"]["sponsorsListing"]["tiers"]["nodes"]

    tiers.map do |t|
      tier = t["monthlyPriceInDollars"]
      raise Error, "Your token needs the 'admin:org' scope to access this API" if t["adminInfo"].nil?

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
    response = open_api("#{API_URL}/repos/#{user}/#{repo}/license")
    return unless response.key?("license")

    response["license"]["spdx_id"]
  rescue HTTPNotFoundError
    nil
  end

  def fetch_pull_requests(name, tap_full_name, state: nil, version: nil)
    if version.present?
      query = "#{name} #{version}"
      regex = /(^|\s)#{Regexp.quote(name)}(:|,|\s)(.*\s)?#{Regexp.quote(version)}(:|,|\s|$)/i
    else
      query = name
      regex = /(^|\s)#{Regexp.quote(name)}(:|,|\s|$)/i
    end
    issues_for_formula(query, tap_full_name: tap_full_name, state: state).select do |pr|
      pr["html_url"].include?("/pull/") && regex.match?(pr["title"])
    end
  rescue RateLimitExceededError => e
    opoo e.message
    []
  end

  def check_for_duplicate_pull_requests(name, tap_full_name, state:, file:, args:, version: nil)
    pull_requests = fetch_pull_requests(name, tap_full_name, state: state, version: version).select do |pr|
      pr_files = open_api(url_to("repos", tap_full_name, "pulls", pr["number"], "files"))
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

  def forked_repo_info!(tap_full_name)
    response = create_fork(tap_full_name)
    # GitHub API responds immediately but fork takes a few seconds to be ready.
    sleep 1 until check_fork_exists(tap_full_name)
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
    tap_full_name = info[:tap_full_name] || tap.full_name
    pr_message = info[:pr_message]

    sourcefile_path.parent.cd do
      git_dir = Utils.popen_read("git rev-parse --git-dir").chomp
      shallow = !git_dir.empty? && File.exist?("#{git_dir}/shallow")
      changed_files = [sourcefile_path]
      changed_files += additional_files if additional_files.present?

      if args.dry_run? || (args.write? && !args.commit?)
        ohai "try to fork repository with GitHub API" unless args.no_fork?
        ohai "git fetch --unshallow origin" if shallow
        ohai "git add #{changed_files.join(" ")}"
        ohai "git checkout --no-track -b #{branch} #{remote}/#{remote_branch}"
        ohai "git commit --no-edit --verbose --message='#{commit_message}'" \
             " -- #{changed_files.join(" ")}"
        ohai "git push --set-upstream $HUB_REMOTE #{branch}:#{branch}"
        ohai "git checkout --quiet #{previous_branch}"
        ohai "create pull request with GitHub API (base branch: #{remote_branch})"
      else

        unless args.commit?
          if args.no_fork?
            remote_url = Utils.popen_read("git remote get-url --push origin").chomp
            username = tap.user
          else
            begin
              remote_url, username = forked_repo_info!(tap_full_name)
            rescue *API_ERRORS => e
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
          url = create_pull_request(tap_full_name, commit_message,
                                    "#{username}:#{branch}", remote_branch, pr_message)["html_url"]
          if args.no_browse?
            puts url
          else
            exec_browser url
          end
        rescue *API_ERRORS => e
          odie "Unable to open pull request: #{e.message}!"
        end
      end
    end
  end

  def pull_request_commits(user, repo, pr, per_page: 100)
    pr_data = open_api(url_to("repos", user, repo, "pulls", pr))
    commits_api = pr_data["commits_url"]
    commit_count = pr_data["commits"]
    commits = []

    if commit_count > API_MAX_ITEMS
      raise Error, "Getting #{commit_count} commits would exceed limit of #{API_MAX_ITEMS} API items!"
    end

    (1..API_MAX_PAGES).each do |page|
      result = open_api(commits_api + "?per_page=#{per_page}&page=#{page}")
      commits.concat(result.map { |c| c["sha"] })

      return commits if commits.length == commit_count

      if result.empty? || page * per_page >= commit_count
        raise Error, "Expected #{commit_count} commits but actually got #{commits.length}!"
      end
    end
  end

  def pull_request_labels(user, repo, pr)
    pr_data = open_api(url_to("repos", user, repo, "pulls", pr))
    pr_data["labels"].map { |label| label["name"] }
  end
end
