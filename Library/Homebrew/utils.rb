# typed: false
# frozen_string_literal: true

require "time"

require "utils/analytics"
require "utils/curl"
require "utils/fork"
require "utils/formatter"
require "utils/gems"
require "utils/git"
require "utils/git_repository"
require "utils/github"
require "utils/gzip"
require "utils/inreplace"
require "utils/link"
require "utils/popen"
require "utils/repology"
require "utils/svn"
require "utils/tty"
require "tap_constants"
require "PATH"
require "extend/kernel"

module Homebrew
  extend Context

  module_function

  def _system(cmd, *args, **options)
    pid = fork do
      yield if block_given?
      args.map!(&:to_s)
      begin
        exec(cmd, *args, **options)
      rescue
        nil
      end
      exit! 1 # never gets here unless exec failed
    end
    Process.wait(T.must(pid))
    $CHILD_STATUS.success?
  end

  def system(cmd, *args, **options)
    if verbose?
      puts "#{cmd} #{args * " "}".gsub(RUBY_PATH, "ruby")
                                 .gsub($LOAD_PATH.join(File::PATH_SEPARATOR).to_s, "$LOAD_PATH")
    end
    _system(cmd, *args, **options)
  end

  # rubocop:disable Style/GlobalVars
  def inject_dump_stats!(the_module, pattern)
    @injected_dump_stat_modules ||= {}
    @injected_dump_stat_modules[the_module] ||= []
    injected_methods = @injected_dump_stat_modules[the_module]
    the_module.module_eval do
      instance_methods.grep(pattern).each do |name|
        next if injected_methods.include? name

        method = instance_method(name)
        define_method(name) do |*args, &block|
          time = Time.now

          begin
            method.bind(self).call(*args, &block)
          ensure
            $times[name] ||= 0
            $times[name] += Time.now - time
          end
        end
      end
    end

    return unless $times.nil?

    $times = {}
    at_exit do
      col_width = [$times.keys.map(&:size).max.to_i + 2, 15].max
      $times.sort_by { |_k, v| v }.each do |method, time|
        puts format("%<method>-#{col_width}s %<time>0.4f sec", method: "#{method}:", time: time)
      end
    end
  end
  # rubocop:enable Style/GlobalVars
end
