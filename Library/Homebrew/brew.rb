# typed: false
# frozen_string_literal: true

if ENV["HOMEBREW_STACKPROF"]
  require_relative "utils/gems"
  Homebrew.setup_gem_environment!
  require "stackprof"
  StackProf.start(mode: :wall, raw: true)
end

raise "HOMEBREW_BREW_FILE was not exported! Please call bin/brew directly!" unless ENV["HOMEBREW_BREW_FILE"]

std_trap = trap("INT") { exit! 130 } # no backtrace thanks

# check ruby version before requiring any modules.
unless ENV["HOMEBREW_REQUIRED_RUBY_VERSION"]
  raise "HOMEBREW_REQUIRED_RUBY_VERSION was not exported! Please call bin/brew directly!"
end

REQUIRED_RUBY_X, REQUIRED_RUBY_Y, = ENV["HOMEBREW_REQUIRED_RUBY_VERSION"].split(".").map(&:to_i)
RUBY_X, RUBY_Y, = RUBY_VERSION.split(".").map(&:to_i)
if RUBY_X < REQUIRED_RUBY_X || (RUBY_X == REQUIRED_RUBY_X && RUBY_Y < REQUIRED_RUBY_Y)
  raise "Homebrew must be run under Ruby #{REQUIRED_RUBY_X}.#{REQUIRED_RUBY_Y}! " \
        "You're running #{RUBY_VERSION}."
end

# Also define here so we can rescue regardless of location.
class MissingEnvironmentVariables < RuntimeError; end

begin
  require_relative "global"
rescue MissingEnvironmentVariables => e
  raise e if ENV["HOMEBREW_MISSING_ENV_RETRY"]

  if ENV["HOMEBREW_DEVELOPER"]
    $stderr.puts <<~EOS
      Warning: #{e.message}
      Retrying with `exec #{ENV["HOMEBREW_BREW_FILE"]}`!
    EOS
  end

  ENV["HOMEBREW_MISSING_ENV_RETRY"] = "1"
  exec ENV["HOMEBREW_BREW_FILE"], *ARGV
end

begin
  trap("INT", std_trap) # restore default CTRL-C handler

  if ENV["CI"]
    $stdout.sync = true
    $stderr.sync = true
  end

  empty_argv = ARGV.empty?
  help_flag_list = %w[-h --help --usage -?]
  help_flag = !ENV["HOMEBREW_HELP"].nil?
  help_cmd_index = nil
  cmd = nil

  ARGV.each_with_index do |arg, i|
    break if help_flag && cmd

    if arg == "help" && !cmd
      # Command-style help: `help <cmd>` is fine, but `<cmd> help` is not.
      help_flag = true
      help_cmd_index = i
    elsif !cmd && help_flag_list.exclude?(arg)
      cmd = ARGV.delete_at(i)
      cmd = Commands::HOMEBREW_INTERNAL_COMMAND_ALIASES.fetch(cmd, cmd)
    end
  end

  ARGV.delete_at(help_cmd_index) if help_cmd_index

  args = Homebrew::CLI::Parser.new.parse(ARGV.dup.freeze, ignore_invalid_options: true)
  Context.current = args.context

  path = PATH.new(ENV["PATH"])
  homebrew_path = PATH.new(ENV["HOMEBREW_PATH"])

  # Add SCM wrappers.
  path.prepend(HOMEBREW_SHIMS_PATH/"scm")
  homebrew_path.prepend(HOMEBREW_SHIMS_PATH/"scm")

  ENV["PATH"] = path

  require "commands"
  require "settings"

  if cmd
    internal_cmd = Commands.valid_internal_cmd?(cmd)
    internal_cmd ||= begin
      internal_dev_cmd = Commands.valid_internal_dev_cmd?(cmd)
      if internal_dev_cmd && !Homebrew::EnvConfig.developer?
        Homebrew::Settings.write "devcmdrun", true
        ENV["HOMEBREW_DEV_CMD_RUN"] = "1"
      end
      internal_dev_cmd
    end
  end

  unless internal_cmd
    # Add contributed commands to PATH before checking.
    homebrew_path.append(Tap.cmd_directories)

    # External commands expect a normal PATH
    ENV["PATH"] = homebrew_path
  end

  # Usage instructions should be displayed if and only if one of:
  # - a help flag is passed AND a command is matched
  # - a help flag is passed AND there is no command specified
  # - no arguments are passed
  # - if cmd is Cask, let Cask handle the help command instead
  if (empty_argv || help_flag) && cmd != "cask"
    require "help"
    Homebrew::Help.help cmd, remaining_args: args.remaining, empty_argv: empty_argv
    # `Homebrew::Help.help` never returns, except for unknown commands.
  end

  if internal_cmd || Commands.external_ruby_v2_cmd_path(cmd)
    Homebrew.send Commands.method_name(cmd)
  elsif (path = Commands.external_ruby_cmd_path(cmd))
    require?(path)
    exit Homebrew.failed? ? 1 : 0
  elsif Commands.external_cmd_path(cmd)
    %w[CACHE LIBRARY_PATH].each do |env|
      ENV["HOMEBREW_#{env}"] = Object.const_get("HOMEBREW_#{env}").to_s
    end
    exec "brew-#{cmd}", *ARGV
  else
    possible_tap = OFFICIAL_CMD_TAPS.find { |_, cmds| cmds.include?(cmd) }
    possible_tap = Tap.fetch(possible_tap.first) if possible_tap

    if !possible_tap || possible_tap.installed? || Tap.untapped_official_taps.include?(possible_tap.name)
      odie "Unknown command: #{cmd}"
    end

    # Unset HOMEBREW_HELP to avoid confusing the tap
    with_env HOMEBREW_HELP: nil do
      tap_commands = []
      cgroup = Utils.popen_read("cat", "/proc/1/cgroup")
      if %w[azpl_job actions_job docker garden kubepods].none? { |container| cgroup.include?(container) }
        brew_uid = HOMEBREW_BREW_FILE.stat.uid
        tap_commands += %W[/usr/bin/sudo -u ##{brew_uid}] if Process.uid.zero? && !brew_uid.zero?
      end
      quiet_arg = args.quiet? ? "--quiet" : nil
      tap_commands += [HOMEBREW_BREW_FILE, "tap", *quiet_arg, possible_tap.name]
      safe_system(*tap_commands)
    end

    ARGV << "--help" if help_flag
    exec HOMEBREW_BREW_FILE, cmd, *ARGV
  end
rescue UsageError => e
  require "help"
  Homebrew::Help.help cmd, remaining_args: args.remaining, usage_error: e.message
rescue SystemExit => e
  onoe "Kernel.exit" if args.debug? && !e.success?
  $stderr.puts e.backtrace if args.debug?
  raise
rescue Interrupt
  $stderr.puts # seemingly a newline is typical
  exit 130
rescue BuildError => e
  Utils::Analytics.report_build_error(e)
  e.dump(verbose: args.verbose?)

  if e.formula.head? || e.formula.deprecated? || e.formula.disabled?
    $stderr.puts <<~EOS
      Please create pull requests instead of asking for help on Homebrew's GitHub,
      Twitter or any other official channels.
    EOS
  end

  exit 1
rescue RuntimeError, SystemCallError => e
  raise if e.message.empty?

  onoe e
  $stderr.puts e.backtrace if args.debug?

  exit 1
rescue MethodDeprecatedError => e
  onoe e
  if e.issues_url
    $stderr.puts "If reporting this issue please do so at (not Homebrew/brew or Homebrew/core):"
    $stderr.puts "  #{Formatter.url(e.issues_url)}"
  end
  $stderr.puts e.backtrace if args.debug?
  exit 1
rescue Exception => e # rubocop:disable Lint/RescueException
  onoe e
  if internal_cmd && defined?(OS::ISSUES_URL) &&
     !Homebrew::EnvConfig.no_auto_update?
    $stderr.puts "#{Tty.bold}Please report this issue:#{Tty.reset}"
    $stderr.puts "  #{Formatter.url(OS::ISSUES_URL)}"
  end
  $stderr.puts e.backtrace
  exit 1
else
  exit 1 if Homebrew.failed?
ensure
  if ENV["HOMEBREW_STACKPROF"]
    StackProf.stop
    StackProf.results("prof/stackprof.dump")
  end
end
