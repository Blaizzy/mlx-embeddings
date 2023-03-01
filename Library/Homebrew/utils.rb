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
  sig { params(the_module: Module, pattern: Regexp).void }
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

  # Converts the array to a comma-separated sentence where the last element is
  # joined by the connector word.
  #
  # You can pass the following options to change the default behavior. If you
  # pass an option key that doesn't exist in the list below, it will raise an
  # <tt>ArgumentError</tt>.
  #
  # ==== Options
  #
  # * <tt>:words_connector</tt> - The sign or word used to join all but the last
  #   element in arrays with three or more elements (default: ", ").
  # * <tt>:last_word_connector</tt> - The sign or word used to join the last element
  #   in arrays with three or more elements (default: ", and ").
  # * <tt>:two_words_connector</tt> - The sign or word used to join the elements
  #   in arrays with two elements (default: " and ").
  #
  # ==== Examples
  #
  #   [].to_sentence                      # => ""
  #   ['one'].to_sentence                 # => "one"
  #   ['one', 'two'].to_sentence          # => "one and two"
  #   ['one', 'two', 'three'].to_sentence # => "one, two, and three"
  #
  #   ['one', 'two'].to_sentence(two_words_connector: '-')
  #   # => "one-two"
  #
  #   ['one', 'two', 'three'].to_sentence(words_connector: ' or ', last_word_connector: ' or at least ')
  #   # => "one or two or at least three"
  sig {
    params(array: T::Array[String], words_connector: String, two_words_connector: String, last_word_connector: String)
      .returns(String)
  }
  def self.to_sentence(array, words_connector: ", ", two_words_connector: " and ", last_word_connector: ", and ")
    case array.length
    when 0
      +""
    when 1
      +(array[0]).to_s
    when 2
      +"#{array[0]}#{two_words_connector}#{array[1]}"
    else
      +"#{array[0...-1].join(words_connector)}#{last_word_connector}#{array[-1]}"
    end
  end
end
