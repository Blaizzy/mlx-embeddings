# typed: strict

class Dependencies < SimpleDelegator
  include Kernel
  # This is a workaround to enable `alias eql? ==`
  # @see https://github.com/sorbet/sorbet/issues/2378#issuecomment-569474238
  sig { params(arg0: BasicObject).returns(T::Boolean) }
  def ==(arg0); end
end

class Requirements < SimpleDelegator
  include Kernel
end
