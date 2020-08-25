# frozen_string_literal: true

old_trap = trap("INT") { exit! 130 }

require "global"
require "extend/ENV"
require "timeout"
require "debrew"
require "formula_assertions"
require "formula_free_port"
require "fcntl"
require "socket"
require "cli/parser"
require "dev-cmd/test"

TEST_TIMEOUT_SECONDS = 5 * 60

begin
  args = Homebrew.test_args.parse
  Context.current = args.context

  error_pipe = UNIXSocket.open(ENV["HOMEBREW_ERROR_PIPE"], &:recv_io)
  error_pipe.fcntl(Fcntl::F_SETFD, Fcntl::FD_CLOEXEC)

  trap("INT", old_trap)

  formula = args.named.to_resolved_formulae.first
  formula.extend(Homebrew::Assertions)
  formula.extend(Homebrew::FreePort)
  formula.extend(Debrew::Formula) if args.debug?

  ENV.extend(Stdenv)
  ENV.setup_build_environment(formula: formula)

  # tests can also return false to indicate failure
  Timeout.timeout TEST_TIMEOUT_SECONDS do
    raise "test returned false" if formula.run_test(keep_tmp: args.keep_tmp?) == false
  end
rescue Exception => e # rubocop:disable Lint/RescueException
  error_pipe.puts e.to_json
  error_pipe.close
ensure
  pid = Process.pid.to_s
  if which("pgrep") && which("pkill") && system("pgrep", "-P", pid, out: :close)
    $stderr.puts "Killing child processes..."
    system "pkill", "-P", pid
    sleep 1
    system "pkill", "-9", "-P", pid
  end
  exit! 1 if e
end
