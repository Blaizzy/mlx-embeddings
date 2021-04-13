#!/usr/bin/env ruby
# frozen_string_literal: true

require "English"

SimpleCov.enable_for_subprocesses true

SimpleCov.start do
  coverage_dir File.expand_path("../test/coverage", File.realpath(__FILE__))
  root File.expand_path("..", File.realpath(__FILE__))

  # We manage the result cache ourselves and the default of 10 minutes can be
  # too low (particularly on Travis CI), causing results from some integration
  # tests to be dropped. This causes random fluctuations in test coverage.
  merge_timeout 86400

  at_fork do |pid|
    # This needs a unique name so it won't be ovewritten
    command_name "#{SimpleCov.command_name} (#{pid})"

    # be quiet, the parent process will be in charge of output and checking coverage totals
    SimpleCov.print_error_status = false
  end
  excludes = ["test", "vendor"]
  subdirs = Dir.chdir(SimpleCov.root) { Pathname.glob("*") }
               .reject { |p| p.extname == ".rb" || excludes.include?(p.to_s) }
               .map { |p| "#{p}/**/*.rb" }.join(",")
  files = "#{SimpleCov.root}/{#{subdirs},*.rb}"

  if ENV["HOMEBREW_INTEGRATION_TEST"]
    # This needs a unique name so it won't be ovewritten
    command_name "#{ENV["HOMEBREW_INTEGRATION_TEST"]} (#{$PROCESS_ID})"

    # be quiet, the parent process will be in charge of output and checking coverage totals
    SimpleCov.print_error_status = false

    SimpleCov.at_exit do
      # Just save result, but don't write formatted output.
      coverage_result = Coverage.result.dup
      Dir[files].each do |file|
        absolute_path = File.expand_path(file)
        coverage_result[absolute_path] ||= SimpleCov::SimulateCoverage.call(absolute_path)
      end
      simplecov_result = SimpleCov::Result.new(coverage_result)
      SimpleCov::ResultMerger.store_result(simplecov_result)

      # If an integration test raises a `SystemExit` exception on exit,
      # exit immediately using the same status code to avoid reporting
      # an error when expecting a non-successful exit status.
      raise if $ERROR_INFO.is_a?(SystemExit)
    end
  else
    command_name "#{command_name} (#{$PROCESS_ID})"

    # Not using this during integration tests makes the tests 4x times faster
    # without changing the coverage.
    track_files files
  end

  add_filter %r{^/build.rb$}
  add_filter %r{^/config.rb$}
  add_filter %r{^/constants.rb$}
  add_filter %r{^/postinstall.rb$}
  add_filter %r{^/test.rb$}
  add_filter %r{^/compat/}
  add_filter %r{^/dev-cmd/tests.rb$}
  add_filter %r{^/test/}
  add_filter %r{^/vendor/}

  require "rbconfig"
  host_os = RbConfig::CONFIG["host_os"]
  add_filter %r{/os/mac} unless /darwin/.match?(host_os)
  add_filter %r{/os/linux} unless /linux/.match?(host_os)

  # Add groups and the proper project name to the output.
  project_name "Homebrew"
  add_group "Cask", %r{^/cask/}
  add_group "Commands", [%r{/cmd/}, %r{^/dev-cmd/}]
  add_group "Extensions", %r{^/extend/}
  add_group "OS", [%r{^/extend/os/}, %r{^/os/}]
  add_group "Requirements", %r{^/requirements/}
  add_group "Scripts", [
    %r{^/brew.rb$},
    %r{^/build.rb$},
    %r{^/postinstall.rb$},
    %r{^/test.rb$},
  ]
end
