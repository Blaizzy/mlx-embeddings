# typed: false
# frozen_string_literal: true

# A formula option.
#
# @api private
class Option
  extend T::Sig

  attr_reader :name, :description, :flag

  def initialize(name, description = "")
    @name = name
    @flag = "--#{name}"
    @description = description
  end

  def to_s
    flag
  end

  def <=>(other)
    return unless other.is_a?(Option)

    name <=> other.name
  end

  def ==(other)
    instance_of?(other.class) && name == other.name
  end
  alias eql? ==

  def hash
    name.hash
  end

  sig { returns(String) }
  def inspect
    "#<#{self.class.name}: #{flag.inspect}>"
  end
end

# A deprecated formula option.
#
# @api private
class DeprecatedOption
  extend T::Sig

  attr_reader :old, :current

  def initialize(old, current)
    @old = old
    @current = current
  end

  sig { returns(String) }
  def old_flag
    "--#{old}"
  end

  sig { returns(String) }
  def current_flag
    "--#{current}"
  end

  def ==(other)
    instance_of?(other.class) && old == other.old && current == other.current
  end
  alias eql? ==
end

# A collection of formula options.
#
# @api private
class Options
  extend T::Sig

  include Enumerable

  def self.create(array)
    new Array(array).map { |e| Option.new(e[/^--([^=]+=?)(.+)?$/, 1] || e) }
  end

  def initialize(*args)
    @options = Set.new(*args)
  end

  def each(*args, &block)
    @options.each(*args, &block)
  end

  def <<(other)
    @options << other
    self
  end

  def +(other)
    self.class.new(@options + other)
  end

  def -(other)
    self.class.new(@options - other)
  end

  def &(other)
    self.class.new(@options & other)
  end

  def |(other)
    self.class.new(@options | other)
  end

  def *(other)
    @options.to_a * other
  end

  def empty?
    @options.empty?
  end

  def as_flags
    map(&:flag)
  end

  def include?(o)
    any? { |opt| opt == o || opt.name == o || opt.flag == o }
  end

  alias to_ary to_a

  sig { returns(String) }
  def inspect
    "#<#{self.class.name}: #{to_a.inspect}>"
  end

  def self.dump_for_formula(f)
    f.options.sort_by(&:flag).each do |opt|
      puts "#{opt.flag}\n\t#{opt.description}"
    end
    puts "--HEAD\n\tInstall HEAD version" if f.head
  end
end
