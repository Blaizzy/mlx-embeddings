# typed: strict
# frozen_string_literal: true

module Utils
  extend T::Sig

  IO_DEFAULT_BUFFER_SIZE = 4096
  private_constant :IO_DEFAULT_BUFFER_SIZE

  sig {
    params(
      arg0:    T.any(String, T::Hash[String, String]),
      args:    String,
      safe:    T::Boolean,
      options: Symbol,
      block:   T.nilable(T.proc.params(io: IO).void),
    ).returns(String)
  }
  def self.popen_read(arg0, *args, safe: false, **options, &block)
    output = popen(args, "rb", options, &block)
    return output if !safe || $CHILD_STATUS.success?

    raise ErrorDuringExecution.new(args, status: $CHILD_STATUS, output: [[:stdout, output]])
  end

  sig {
    params(
      arg0:    T.any(String, T::Hash[String, String]),
      args:    String,
      options: Symbol,
      block:   T.nilable(T.proc.params(io: IO).void),
    ).returns(String)
  }
  def self.safe_popen_read(arg0, *args, **options, &block)
    popen_read(*args, safe: true, **options, &block)
  end

  sig {
    params(
      arg0:    T.any(String, T::Hash[String, String]),
      args:    String,
      safe:    T::Boolean,
      options: Symbol,
      _block:  T.proc.params(io: IO).void,
    ).returns(String)
  }
  def self.popen_write(arg0, *args, safe: false, **options, &_block)
    output = ""
    popen(args, "w+b", options) do |pipe|
      # Before we yield to the block, capture as much output as we can
      loop do
        output += pipe.read_nonblock(IO_DEFAULT_BUFFER_SIZE)
      rescue IO::WaitReadable, EOFError
        break
      end

      yield pipe
      pipe.close_write
      pipe.wait_readable

      # Capture the rest of the output
      output += T.must(pipe.read)
      output.freeze
    end
    return output if !safe || $CHILD_STATUS.success?

    raise ErrorDuringExecution.new(args, status: $CHILD_STATUS, output: [[:stdout, output]])
  end

  sig {
    params(
      arg0:    T.any(String, T::Hash[String, String]),
      args:    String,
      options: Symbol,
      block:   T.nilable(T.proc.params(io: IO).void),
    ).returns(String)
  }
  def self.safe_popen_write(arg0, *args, **options, &block)
    popen_write(*args, safe: true, **options, &block)
  end

  sig {
    params(
      args:    T::Array[T.any(String, T::Hash[String, String])],
      mode:    String,
      options: T::Hash[Symbol, T.untyped],
      block:   T.nilable(T.proc.params(io: IO).void),
    ).returns(String)
  }
  def self.popen(args, mode, options = {}, &block)
    IO.popen("-", mode) do |pipe|
      if pipe
        return pipe.read unless block

        yield pipe
      else
        options[:err] ||= "/dev/null" unless ENV["HOMEBREW_STDERR"]
        begin
          exec(*args, options)
        rescue Errno::ENOENT
          $stderr.puts "brew: command not found: #{args[0]}" unless options[:err] == :close
          exit! 127
        rescue SystemCallError
          $stderr.puts "brew: exec failed: #{args[0]}" unless options[:err] == :close
          exit! 1
        end
      end
    end
  end
end
