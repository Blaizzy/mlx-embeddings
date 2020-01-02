# frozen_string_literal: true

require "formula"
require "install"
require "system_config"
require "stringio"
require "socket"
require "cli/parser"

module Homebrew
  module_function

  def gist_logs_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `gist-logs` [<options>] <formula>

        Upload logs for a failed build of <formula> to a new Gist. Presents an
        error message if no logs are found.
      EOS
      switch "--with-hostname",
             description: "Include the hostname in the Gist."
      switch "-n", "--new-issue",
             description: "Automatically create a new issue in the appropriate GitHub repository "\
                          "after creating the Gist."
      switch "-p", "--private",
             description: "The Gist will be marked private and will not appear in listings but will "\
                          "be accessible with its link."
      switch :verbose
      switch :debug
      max_named 1
    end
  end

  def gistify_logs(f)
    gist_logs_args.parse

    files = load_logs(f.logs)
    build_time = f.logs.ctime
    timestamp = build_time.strftime("%Y-%m-%d_%H-%M-%S")

    s = StringIO.new
    SystemConfig.dump_verbose_config s
    # Dummy summary file, asciibetically first, to control display title of gist
    files["# #{f.name} - #{timestamp}.txt"] = { content: brief_build_info(f) }
    files["00.config.out"] = { content: s.string }
    files["00.doctor.out"] = { content: Utils.popen_read("#{HOMEBREW_PREFIX}/bin/brew", "doctor", err: :out) }
    unless f.core_formula?
      tap = <<~EOS
        Formula: #{f.name}
            Tap: #{f.tap}
           Path: #{f.path}
      EOS
      files["00.tap.out"] = { content: tap }
    end

    if GitHub.api_credentials_type == :none
      puts <<~EOS
        You can create a new personal access token:
          #{GitHub::ALL_SCOPES_URL}
        #{Utils::Shell.set_variable_in_profile("HOMEBREW_GITHUB_API_TOKEN", "your_token_here")}

      EOS
      login!
    end

    # Description formatted to work well as page title when viewing gist
    if f.core_formula?
      descr = "#{f.name} on #{OS_VERSION} - Homebrew build logs"
    else
      descr = "#{f.name} (#{f.full_name}) on #{OS_VERSION} - Homebrew build logs"
    end
    url = create_gist(files, descr)

    url = create_issue(f.tap, "#{f.name} failed to build on #{MacOS.full_version}", url) if args.new_issue?

    puts url if url
  end

  def brief_build_info(f)
    build_time_str = f.logs.ctime.strftime("%Y-%m-%d %H:%M:%S")
    s = +<<~EOS
      Homebrew build logs for #{f.full_name} on #{OS_VERSION}
    EOS
    if args.with_hostname?
      hostname = Socket.gethostname
      s << "Host: #{hostname}\n"
    end
    s << "Build date: #{build_time_str}\n"
    s.freeze
  end

  # Causes some terminals to display secure password entry indicators
  def noecho_gets
    system "stty -echo"
    result = $stdin.gets
    system "stty echo"
    puts
    result
  end

  def login!
    print "GitHub User: "
    ENV["HOMEBREW_GITHUB_API_USERNAME"] = $stdin.gets.chomp
    print "Password: "
    ENV["HOMEBREW_GITHUB_API_PASSWORD"] = noecho_gets.chomp
    puts
  end

  def load_logs(dir)
    logs = {}
    if dir.exist?
      dir.children.sort.each do |file|
        contents = file.size? ? file.read : "empty log"
        # small enough to avoid GitHub "unicorn" page-load-timeout errors
        max_file_size = 1_000_000
        contents = truncate_text_to_approximate_size(contents, max_file_size, front_weight: 0.2)
        logs[file.basename.to_s] = { content: contents }
      end
    end
    raise "No logs." if logs.empty?

    logs
  end

  def create_private?
    args.private?
  end

  def create_gist(files, description)
    url = "https://api.github.com/gists"
    data = { "public" => !create_private?, "files" => files, "description" => description }
    scopes = GitHub::CREATE_GIST_SCOPES
    GitHub.open_api(url, data: data, scopes: scopes)["html_url"]
  end

  def create_issue(repo, title, body)
    url = "https://api.github.com/repos/#{repo}/issues"
    data = { "title" => title, "body" => body }
    scopes = GitHub::CREATE_ISSUE_FORK_OR_PR_SCOPES
    GitHub.open_api(url, data: data, scopes: scopes)["html_url"]
  end

  def gist_logs
    raise FormulaUnspecifiedError if Homebrew.args.resolved_formulae.length != 1

    Install.perform_preinstall_checks(all_fatal: true)
    Install.perform_build_from_source_checks(all_fatal: true)
    gistify_logs(Homebrew.args.resolved_formulae.first)
  end
end
