# frozen_string_literal: true

old_trap = trap("INT") { exit! 130 }

require "global"
require "extend/ENV"
require "timeout"
require "debrew"
require "formula_assertions"
require "fcntl"
require "socket"
require "cli/parser"

def test_args
  Homebrew::CLI::Parser.new do
    switch :force
    switch :verbose
    switch :debug
  end
end

TEST_TIMEOUT_SECONDS = 5 * 60

begin
  test_args.parse
  error_pipe = UNIXSocket.open(ENV["HOMEBREW_ERROR_PIPE"], &:recv_io)
  error_pipe.fcntl(Fcntl::F_SETFD, Fcntl::FD_CLOEXEC)

  ENV.extend(Stdenv)
  ENV.setup_build_environment

  trap("INT", old_trap)

  formula = Homebrew.args.resolved_formulae.first
  formula.extend(Homebrew::Assertions)
  formula.extend(Debrew::Formula) if Homebrew.args.debug?

  # tests can also return false to indicate failure
  Timeout.timeout TEST_TIMEOUT_SECONDS do
    raise "test returned false" if formula.run_test == false
  end
rescue Exception => e # rubocop:disable Lint/RescueException
  error_pipe.puts e.to_json
  error_pipe.close
  pid = Process.pid.to_s
  if which("pgrep") && which("pkill") && system("pgrep", "-P", pid, out: :close)
    $stderr.puts "Killing child processes..."
    system "pkill", "-P", pid
    sleep 1
    system "pkill", "-9", "-P", pid
  end
  exit! 1
end
