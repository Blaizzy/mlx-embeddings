# typed: false
# frozen_string_literal: true

require "cli/parser"
require "fileutils"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def tests_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Run Homebrew's unit and integration tests.
      EOS
      switch "--coverage",
             description: "Generate code coverage reports."
      switch "--generic",
             description: "Run only OS-agnostic tests."
      switch "--no-compat",
             description: "Do not load the compatibility layer when running tests."
      switch "--online",
             description: "Include tests that use the GitHub API and tests that use any of the taps for " \
                          "official external commands."
      switch "--byebug",
             description: "Enable debugging using byebug."
      switch "--changed",
             description: "Only runs tests on files that were changed from the master branch."
      flag   "--only=",
             description: "Run only <test_script>`_spec.rb`. Appending `:`<line_number> will start at a " \
                          "specific line."
      flag   "--seed=",
             description: "Randomise tests with the specified <value> instead of a random seed."

      conflicts "--changed", "--only"

      named_args :none
    end
  end

  def use_buildpulse?
    return @use_buildpulse if defined?(@use_buildpulse)

    @use_buildpulse = ENV["HOMEBREW_BUILDPULSE_ACCESS_KEY_ID"].present? &&
                      ENV["HOMEBREW_BUILDPULSE_SECRET_ACCESS_KEY"].present? &&
                      ENV["HOMEBREW_BUILDPULSE_ACCOUNT_ID"].present? &&
                      ENV["HOMEBREW_BUILDPULSE_REPOSITORY_ID"].present?
  end

  def run_buildpulse
    require "formula"

    with_env(HOMEBREW_NO_AUTO_UPDATE: "1", HOMEBREW_NO_BOOTSNAP: "1") do
      ensure_formula_installed!("buildpulse-test-reporter",
                                reason: "reporting test flakiness")
    end

    ENV["BUILDPULSE_ACCESS_KEY_ID"] = ENV.fetch("HOMEBREW_BUILDPULSE_ACCESS_KEY_ID")
    ENV["BUILDPULSE_SECRET_ACCESS_KEY"] = ENV.fetch("HOMEBREW_BUILDPULSE_SECRET_ACCESS_KEY")

    ohai "Sending test results to BuildPulse"

    safe_system Formula["buildpulse-test-reporter"].opt_bin/"buildpulse-test-reporter",
                "submit", "#{HOMEBREW_LIBRARY_PATH}/test/junit",
                "--account-id", ENV.fetch("HOMEBREW_BUILDPULSE_ACCOUNT_ID"),
                "--repository-id", ENV.fetch("HOMEBREW_BUILDPULSE_REPOSITORY_ID")
  end

  def changed_test_files
    changed_files = Utils.popen_read("git", "diff", "--name-only", "master")

    raise UsageError, "No files have been changed from the master branch!" if changed_files.blank?

    filestub_regex = %r{Library/Homebrew/([\w/-]+).rb}
    changed_files.scan(filestub_regex).map(&:last).map do |filestub|
      if filestub.start_with?("test/")
        # Only run tests on *_spec.rb files in test/ folder
        filestub.end_with?("_spec") ? Pathname("#{filestub}.rb") : nil
      else
        # For all other changed .rb files guess the associated test file name
        Pathname("test/#{filestub}_spec.rb")
      end
    end.compact.select(&:exist?)
  end

  def tests
    args = tests_args.parse

    Homebrew.install_bundler_gems!

    require "byebug" if args.byebug?

    HOMEBREW_LIBRARY_PATH.cd do
      # Cleanup any unwanted user configuration.
      allowed_test_env = %w[
        HOMEBREW_GITHUB_API_TOKEN
        HOMEBREW_CACHE
        HOMEBREW_LOGS
        HOMEBREW_TEMP
      ]
      Homebrew::EnvConfig::ENVS.keys.map(&:to_s).each do |env|
        next if allowed_test_env.include?(env)

        ENV.delete(env)
      end

      ENV["HOMEBREW_NO_ANALYTICS_THIS_RUN"] = "1"
      ENV["HOMEBREW_NO_COMPAT"] = "1" if args.no_compat?
      ENV["HOMEBREW_TEST_GENERIC_OS"] = "1" if args.generic?
      ENV["HOMEBREW_TEST_ONLINE"] = "1" if args.online?
      ENV["HOMEBREW_SORBET_RUNTIME"] = "1"

      ENV["USER"] ||= system_command!("id", args: ["-nu"]).stdout.chomp

      # Avoid local configuration messing with tests, e.g. git being configured
      # to use GPG to sign by default
      ENV["HOME"] = "#{HOMEBREW_LIBRARY_PATH}/test"

      # Print verbose output when requesting debug or verbose output.
      ENV["HOMEBREW_VERBOSE_TESTS"] = "1" if args.debug? || args.verbose?

      if args.coverage?
        ENV["HOMEBREW_TESTS_COVERAGE"] = "1"
        FileUtils.rm_f "test/coverage/.resultset.json"
      end

      # Override author/committer as global settings might be invalid and thus
      # will cause silent failure during the setup of dummy Git repositories.
      %w[AUTHOR COMMITTER].each do |role|
        ENV["GIT_#{role}_NAME"] = "brew tests"
        ENV["GIT_#{role}_EMAIL"] = "brew-tests@localhost"
        ENV["GIT_#{role}_DATE"]  = "Sun Jan 22 19:59:13 2017 +0000"
      end

      parallel = true

      files = if args.only
        test_name, line = args.only.split(":", 2)

        if line.nil?
          Dir.glob("test/{#{test_name},#{test_name}/**/*}_spec.rb")
        else
          parallel = false
          ["test/#{test_name}_spec.rb:#{line}"]
        end
      elsif args.changed?
        changed_test_files
      else
        Dir.glob("test/**/*_spec.rb")
      end

      if files.blank?
        raise UsageError, "The --only= argument requires a valid file or folder name!" if args.only

        if args.changed?
          opoo "No tests are directly associated with the changed files!"
          return
        end
      end

      parallel_rspec_log_name = "parallel_runtime_rspec"
      parallel_rspec_log_name = "#{parallel_rspec_log_name}.no_compat" if args.no_compat?
      parallel_rspec_log_name = "#{parallel_rspec_log_name}.generic" if args.generic?
      parallel_rspec_log_name = "#{parallel_rspec_log_name}.online" if args.online?
      parallel_rspec_log_name = "#{parallel_rspec_log_name}.log"

      parallel_rspec_log_path = if ENV["CI"]
        "tests/#{parallel_rspec_log_name}"
      else
        "#{HOMEBREW_CACHE}/#{parallel_rspec_log_name}"
      end
      ENV["PARALLEL_RSPEC_LOG_PATH"] = parallel_rspec_log_path

      parallel_args = if ENV["CI"]
        %W[
          --combine-stderr
          --serialize-stdout
          --runtime-log #{parallel_rspec_log_path}
        ]
      else
        %w[
          --nice
        ]
      end

      # Generate seed ourselves and output later to avoid multiple different
      # seeds being output when running parallel tests.
      seed = args.seed || rand(0xFFFF).to_i

      bundle_args = ["-I", HOMEBREW_LIBRARY_PATH/"test"]
      bundle_args += %W[
        --seed #{seed}
        --color
        --require spec_helper
      ]

      unless OS.mac?
        bundle_args << "--tag" << "~needs_macos" << "--tag" << "~cask"
        files = files.grep_v(%r{^test/(os/mac|cask)(/.*|_spec\.rb)$})
      end

      unless OS.linux?
        bundle_args << "--tag" << "~needs_linux"
        files = files.grep_v(%r{^test/os/linux(/.*|_spec\.rb)$})
      end

      puts "Randomized with seed #{seed}"

      # Let tests find `bundle` in the actual location.
      ENV["HOMEBREW_TESTS_GEM_USER_DIR"] = gem_user_dir

      # Let `bundle` in PATH find its gem.
      ENV["GEM_PATH"] = "#{ENV.fetch("GEM_PATH")}:#{gem_user_dir}"

      # Submit test flakiness information using BuildPulse
      # BUILDPULSE used in spec_helper.rb
      if use_buildpulse?
        ENV["BUILDPULSE"] = "1"
        ohai "Running tests with BuildPulse-friendly settings"
      end

      if parallel
        system "bundle", "exec", "parallel_rspec", *parallel_args, "--", *bundle_args, "--", *files
      else
        system "bundle", "exec", "rspec", *bundle_args, "--", *files
      end

      run_buildpulse if use_buildpulse?

      return if $CHILD_STATUS.success?

      Homebrew.failed = true
    end
  end
end
