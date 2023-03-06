# typed: true
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
  extend T::Sig

  def self._system(cmd, *args, **options)
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

  def self.system(cmd, *args, **options)
    if verbose?
      puts "#{cmd} #{args * " "}".gsub(RUBY_PATH, "ruby")
                                 .gsub($LOAD_PATH.join(File::PATH_SEPARATOR).to_s, "$LOAD_PATH")
    end
    _system(cmd, *args, **options)
  end

  # rubocop:disable Style/GlobalVars
  sig { params(the_module: Module, pattern: Regexp).void }
  def self.inject_dump_stats!(the_module, pattern)
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

module Utils
  extend T::Sig

  # Removes the rightmost segment from the constant expression in the string.
  #
  #   deconstantize('Net::HTTP')   # => "Net"
  #   deconstantize('::Net::HTTP') # => "::Net"
  #   deconstantize('String')      # => ""
  #   deconstantize('::String')    # => ""
  #   deconstantize('')            # => ""
  #
  # See also #demodulize.
  # @see https://github.com/rails/rails/blob/b0dd7c7/activesupport/lib/active_support/inflector/methods.rb#L247-L258
  #   `ActiveSupport::Inflector.deconstantize`
  sig { params(path: String).returns(String) }
  def self.deconstantize(path)
    T.must(path[0, path.rindex("::") || 0]) # implementation based on the one in facets' Module#spacename
  end

  # Removes the module part from the expression in the string.
  #
  #   demodulize('ActiveSupport::Inflector::Inflections') # => "Inflections"
  #   demodulize('Inflections')                           # => "Inflections"
  #   demodulize('::Inflections')                         # => "Inflections"
  #   demodulize('')                                      # => ""
  #
  # See also #deconstantize.
  # @see https://github.com/rails/rails/blob/b0dd7c7/activesupport/lib/active_support/inflector/methods.rb#L230-L245
  #   `ActiveSupport::Inflector.demodulize`
  sig { params(path: String).returns(String) }
  def self.demodulize(path)
    if (i = path.rindex("::"))
      T.must(path[(i + 2)..])
    else
      path
    end
  end

  # A lightweight alternative to `ActiveSupport::Inflector.pluralize`:
  # Combines `stem` with the `singular` or `plural` suffix based on `count`.
  sig { params(stem: String, count: Integer, plural: String, singular: String).returns(String) }
  def self.pluralize(stem, count, plural: "s", singular: "")
    suffix = (count == 1) ? singular : plural
    "#{stem}#{suffix}"
  end
end
