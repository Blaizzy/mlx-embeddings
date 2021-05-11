# typed: true
# frozen_string_literal: true

require "open3"
require "ostruct"
require "plist"
require "shellwords"

require "extend/io"
require "extend/predicable"
require "extend/hash_validator"

require "extend/time"

# Class for running sub-processes and capturing their output and exit status.
#
# @api private
class SystemCommand
  extend T::Sig

  using TimeRemaining

  # Helper functions for calling {SystemCommand.run}.
  module Mixin
    extend T::Sig

    def system_command(*args)
      T.unsafe(SystemCommand).run(*args)
    end

    def system_command!(*args)
      T.unsafe(SystemCommand).run!(*args)
    end
  end

  include Context
  extend Predicable

  def self.run(executable, **options)
    T.unsafe(self).new(executable, **options).run!
  end

  def self.run!(command, **options)
    T.unsafe(self).run(command, **options, must_succeed: true)
  end

  sig { returns(SystemCommand::Result) }
  def run!
    $stderr.puts redact_secrets(command.shelljoin.gsub('\=', "="), @secrets) if verbose? || debug?

    @output = []

    each_output_line do |type, line|
      case type
      when :stdout
        $stdout << line if print_stdout?
        @output << [:stdout, line]
      when :stderr
        $stderr << line if print_stderr?
        @output << [:stderr, line]
      end
    end

    result = Result.new(command, @output, @status, secrets: @secrets)
    result.assert_success! if must_succeed?
    result
  end

  sig {
    params(
      executable:   T.any(String, Pathname),
      args:         T::Array[T.any(String, Integer, Float, URI::Generic)],
      sudo:         T::Boolean,
      env:          T::Hash[String, String],
      input:        T.any(String, T::Array[String]),
      must_succeed: T::Boolean,
      print_stdout: T::Boolean,
      print_stderr: T::Boolean,
      debug:        T.nilable(T::Boolean),
      verbose:      T.nilable(T::Boolean),
      secrets:      T.any(String, T::Array[String]),
      chdir:        T.any(String, Pathname),
      timeout:      T.nilable(T.any(Integer, Float)),
    ).void
  }
  def initialize(
    executable,
    args: [],
    sudo: false,
    env: {},
    input: [],
    must_succeed: false,
    print_stdout: false,
    print_stderr: true,
    debug: nil,
    verbose: false,
    secrets: [],
    chdir: T.unsafe(nil),
    timeout: nil
  )
    require "extend/ENV"
    @executable = executable
    @args = args
    @sudo = sudo
    env.each_key do |name|
      next if /^[\w&&\D]\w*$/.match?(name)

      raise ArgumentError, "Invalid variable name: #{name}"
    end
    @env = env
    @input = Array(input)
    @must_succeed = must_succeed
    @print_stdout = print_stdout
    @print_stderr = print_stderr
    @debug = debug
    @verbose = verbose
    @secrets = (Array(secrets) + ENV.sensitive_environment.values).uniq
    @chdir = chdir
    @timeout = timeout
  end

  sig { returns(T::Array[String]) }
  def command
    [*sudo_prefix, *env_args, executable.to_s, *expanded_args]
  end

  private

  attr_reader :executable, :args, :input, :chdir, :env

  attr_predicate :sudo?, :print_stdout?, :print_stderr?, :must_succeed?

  sig { returns(T::Boolean) }
  def debug?
    return super if @debug.nil?

    @debug
  end

  sig { returns(T::Boolean) }
  def verbose?
    return super if @verbose.nil?

    @verbose
  end

  sig { returns(T::Array[String]) }
  def env_args
    set_variables = env.compact.map do |name, value|
      sanitized_name = Shellwords.escape(name)
      sanitized_value = Shellwords.escape(value)
      "#{sanitized_name}=#{sanitized_value}"
    end

    return [] if set_variables.empty?

    ["/usr/bin/env", *set_variables]
  end

  sig { returns(T::Array[String]) }
  def sudo_prefix
    return [] unless sudo?

    askpass_flags = ENV.key?("SUDO_ASKPASS") ? ["-A"] : []
    ["/usr/bin/sudo", *askpass_flags, "-E", "--"]
  end

  sig { returns(T::Array[String]) }
  def expanded_args
    @expanded_args ||= args.map do |arg|
      if arg.respond_to?(:to_path)
        File.absolute_path(arg)
      elsif arg.is_a?(Integer) || arg.is_a?(Float) || arg.is_a?(URI::Generic)
        arg.to_s
      else
        arg.to_str
      end
    end
  end

  class ProcessTerminatedInterrupt < StandardError; end
  private_constant :ProcessTerminatedInterrupt

  sig { params(block: T.proc.params(type: Symbol, line: String).void).void }
  def each_output_line(&block)
    executable, *args = command
    options = {
      # Create a new process group so that we can send `SIGINT` from
      # parent to child rather than the child receiving `SIGINT` directly.
      pgroup: sudo? ? nil : true,
    }
    options[:chdir] = chdir if chdir

    pid = T.let(nil, T.nilable(Integer))
    raw_stdin, raw_stdout, raw_stderr, raw_wait_thr = ignore_interrupts do
      T.unsafe(Open3).popen3(env, [executable, executable], *args, **options)
       .tap { |*, wait_thr| pid = wait_thr.pid }
    end

    write_input_to(raw_stdin)
    raw_stdin.close_write

    thread_ready_queue = Queue.new
    thread_done_queue = Queue.new
    line_thread = Thread.new do
      Thread.handle_interrupt(ProcessTerminatedInterrupt => :never) do
        thread_ready_queue << true
        each_line_from [raw_stdout, raw_stderr], &block
      end
      thread_done_queue.pop
    rescue ProcessTerminatedInterrupt
      nil
    end

    end_time = Time.now + @timeout if @timeout
    raise Timeout::Error if raw_wait_thr.join(end_time&.remaining).nil?

    @status = raw_wait_thr.value

    thread_ready_queue.pop
    line_thread.raise ProcessTerminatedInterrupt.new
    thread_done_queue << true
    line_thread.join
  rescue Interrupt
    Process.kill("INT", pid) if pid && !sudo?
    raise Interrupt
  rescue SystemCallError => e
    @status = $CHILD_STATUS
    @output << [:stderr, e.message]
  end

  sig { params(raw_stdin: IO).void }
  def write_input_to(raw_stdin)
    input.each(&raw_stdin.method(:write))
  end

  sig { params(sources: T::Array[IO], _block: T.proc.params(type: Symbol, line: String).void).void }
  def each_line_from(sources, &_block)
    sources = {
      sources[0] => :stdout,
      sources[1] => :stderr,
    }

    pending_interrupt = T.let(false, T::Boolean)

    until pending_interrupt
      readable_sources = T.let([], T::Array[IO])
      begin
        Thread.handle_interrupt(ProcessTerminatedInterrupt => :on_blocking) do
          readable_sources = T.must(IO.select(sources.keys)).fetch(0)
        end
      rescue ProcessTerminatedInterrupt
        readable_sources = sources.keys
        pending_interrupt = true
      end

      break if readable_sources.none? do |source|
        loop do
          line = source.readline_nonblock || ""
          yield(sources.fetch(source), line)
        end
      rescue EOFError
        source.close_read
        sources.delete(source)
        sources.any?
      rescue IO::WaitReadable
        true
      end
    end

    sources.each_key(&:close_read)
  end

  # Result containing the output and exit status of a finished sub-process.
  class Result
    extend T::Sig

    include Context

    attr_accessor :command, :status, :exit_status

    sig {
      params(
        command: T::Array[String],
        output:  T::Array[[Symbol, String]],
        status:  Process::Status,
        secrets: T::Array[String],
      ).void
    }
    def initialize(command, output, status, secrets:)
      @command       = command
      @output        = output
      @status        = status
      @exit_status   = status.exitstatus
      @secrets       = secrets
    end

    sig { void }
    def assert_success!
      return if @status.success?

      raise ErrorDuringExecution.new(command, status: @status, output: @output, secrets: @secrets)
    end

    sig { returns(String) }
    def stdout
      @stdout ||= @output.select { |type,| type == :stdout }
                         .map { |_, line| line }
                         .join
    end

    sig { returns(String) }
    def stderr
      @stderr ||= @output.select { |type,| type == :stderr }
                         .map { |_, line| line }
                         .join
    end

    sig { returns(String) }
    def merged_output
      @merged_output ||= @output.map { |_, line| line }
                                .join
    end

    sig { returns(T::Boolean) }
    def success?
      return false if @exit_status.nil?

      @exit_status.zero?
    end

    sig { returns([String, String, Process::Status]) }
    def to_ary
      [stdout, stderr, status]
    end

    sig { returns(T.nilable(T.any(Array, Hash))) }
    def plist
      @plist ||= begin
        output = stdout

        output = output.sub(/\A(.*?)(\s*<\?\s*xml)/m) do
          warn_plist_garbage(T.must(Regexp.last_match(1)))
          Regexp.last_match(2)
        end

        output = output.sub(%r{(<\s*/\s*plist\s*>\s*)(.*?)\Z}m) do
          warn_plist_garbage(T.must(Regexp.last_match(2)))
          Regexp.last_match(1)
        end

        Plist.parse_xml(output)
      end
    end

    sig { params(garbage: String).void }
    def warn_plist_garbage(garbage)
      return unless verbose?
      return unless garbage.match?(/\S/)

      opoo "Received non-XML output from #{Formatter.identifier(command.first)}:"
      $stderr.puts garbage.strip
    end
    private :warn_plist_garbage
  end
end

# Make `system_command` available everywhere.
# FIXME: Include this explicitly only where it is needed.
include SystemCommand::Mixin # rubocop:disable Style/MixinUsage
