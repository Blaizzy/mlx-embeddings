# frozen_string_literal: true

# Options for a formula build.
#
# @api private
class BuildOptions
  # @private
  def initialize(args, options)
    @args = args
    @options = options
  end

  # TODO: rename private_include? when include? is removed.
  # @deprecated
  def include?(name)
    odeprecated "BuildOptions#include?"
    private_include?("--#{name}")
  end

  # True if a {Formula} is being built with a specific option.
  # <pre>args << "--i-want-spam" if build.with? "spam"
  #
  # args << "--qt-gui" if build.with? "qt" # "--with-qt" ==> build.with? "qt"
  #
  # # If a formula presents a user with a choice, but the choice must be fulfilled:
  # if build.with? "example2"
  #   args << "--with-example2"
  # else
  #   args << "--with-example1"
  # end</pre>
  def with?(val)
    option_names = val.respond_to?(:option_names) ? val.option_names : [val]

    option_names.any? do |name|
      if option_defined? "with-#{name}"
        private_include? "with-#{name}"
      elsif option_defined? "without-#{name}"
        !private_include? "without-#{name}"
      else
        false
      end
    end
  end

  # True if a {Formula} is being built without a specific option.
  # <pre>args << "--no-spam-plz" if build.without? "spam"</pre>
  def without?(val)
    !with?(val)
  end

  # True if a {Formula} is being built as a bottle (i.e. binary package).
  def bottle?
    private_include? "build-bottle"
  end

  # True if a {Formula} is being built with {Formula.head} instead of {Formula.stable}.
  # <pre>args << "--some-new-stuff" if build.head?</pre>
  # <pre># If there are multiple conditional arguments use a block instead of lines.
  #  if build.head?
  #    args << "--i-want-pizza"
  #    args << "--and-a-cold-beer" if build.with? "cold-beer"
  #  end</pre>
  def head?
    private_include? "HEAD"
  end

  # @private
  def devel?
    odisabled "BuildOptions#devel?"
  end

  # True if a {Formula} is being built with {Formula.stable} instead of {Formula.head}.
  # This is the default.
  # <pre>args << "--some-beta" if build.head?</pre>
  def stable?
    !head?
  end

  # True if a {Formula} is being built universally.
  # e.g. on newer Intel Macs this means a combined x86_64/x86 binary/library.
  # <pre>args << "--universal-binary" if build.universal?</pre>
  def universal?
    odeprecated "BuildOptions#universal?"
    private_include?("universal") && option_defined?("universal")
  end

  # True if a {Formula} is being built in C++11 mode.
  def cxx11?
    odeprecated "BuildOptions#cxx11?"
    private_include?("c++11") && option_defined?("c++11")
  end

  # True if the build has any arguments or options specified.
  def any_args_or_options?
    !@args.empty? || !@options.empty?
  end

  # @private
  def used_options
    @options & @args
  end

  # @private
  def unused_options
    @options - @args
  end

  private

  # TODO: rename to include? when include? is removed.
  def private_include?(name)
    @args.include?("--#{name}")
  end

  def option_defined?(name)
    @options.include? name
  end
end
